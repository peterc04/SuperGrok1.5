#ifndef GROKKING_KERNELS_GFX942_SUPERGROK2_GFX942_HIP_HPP_
#define GROKKING_KERNELS_GFX942_SUPERGROK2_GFX942_HIP_HPP_
// ============================================================================
// supergrok2_gfx942.hip.hpp — CANONICAL SuperGrok gfx942 step logic for 'supergrok2'.
//
// AMDGCN-asm status: NOT PRESENT. This path is ATen + rocBLAS
// (rocBLAS dispatches MFMA v_mfma_f32_16x16x16_bf16 internally for
// BF16/FP16 GEMMs >= 16). Native __global__ + inline AMDGCN asm
// requires migrating this file from .hip.cpp to .hip (hipcc-routed);
// tracked as roadmap item 2. These .hip.cpp TUs route through the host
// compiler (g++/clang++), which is why they hold host ATen orchestration
// rather than device kernels.
//
// The production TU csrc/backends/hip/gfx942/launch_supergrok2.hip.cpp now
// #include's this header and keeps only the host launcher(s) the pybind
// layer calls. Migrated byte-for-byte from that TU.
// ============================================================================
// HIP gfx942 launch glue for SuperGrok v2 (CSA/HCA meta-model).
// Algorithm: csrc/algorithms/supergrok2.h
//
// HONEST STATUS: Functional ATen port of the DeepSeek-V4-style CSA/HCA hybrid
// attention meta-model that replaced the bidirectional Mamba-3 selective scan.
// Math mirrors the sm_90 Hopper path semantically:
//
//   CSA (Compressed Sparse Attention, compression m=4): strided weighted KV
//   pooling, a low-rank "lightning indexer" that scores compressed entries and
//   keeps the top-k, plus a causal sliding window, then multi-head
//   softmax(QKᵀ/√head_dim)·V → csa_ctx [N, d_model]. This is the fine-grained /
//   local context that previously came from mamba_fwd.
//
//   HCA (Heavily Compressed Attention, compression m'=128): stride-128 mean
//   pool, dense attention over ALL compressed entries plus the sliding window →
//   hca_ctx [N, d_model]. This is the global coarse context that previously came
//   from mamba_bwd.
//
// The GRU + PEER product-key routing + per-element expert MLP + AdamW apply tail
// are KEPT VERBATIM from the Mamba-3 era (spec §3b): only the sequence mixer
// (scan) and its weight set changed. Attention is stateless across optimizer
// steps, so the launcher drops the mamba_fwd/bwd carried states; the GRU state
// is still carried.
//
//   • PyTorch routes `.hip.cpp` files through the HOST compiler (g++/clang++),
//     not hipcc. We therefore cannot define `__global__` kernels or use
//     `<<<...>>>` launch syntax here. ALL work goes through ATen tensor ops,
//     which dispatch to rocBLAS / rocPRIM. Projection GEMMs (input_proj, q/k/v,
//     indexer, attn-out, expert MLP, GRU linears) reach MFMA via rocBLAS
//     internally (BF16/FP16 input, FP32 accumulate). Attention QKᵀ / P·V go
//     through torch::matmul; softmax/top-k/gather go through ATen.
//
//   • The bilevel backward path is NOT implemented on gfx942 (it throws). The
//     forward step / batched step is functional.
//
// Build matrix: SG2 / gfx942 stays 🟡 (compiles, math is correct, perf not
// hardware-verified).

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "csrc/algorithms/supergrok2_bilevel_adjoint.h"

// ── Autotuner-consumable launch parameters (inlined; see compile.py) ──
// (formerly csrc/tuning.h — folded in per spec §9/§10)
#ifndef SG_TUNED_BLOCK_SIZE
#define SG_TUNED_BLOCK_SIZE 256
#endif
#ifndef SG_TUNED_VEC_WIDTH
#define SG_TUNED_VEC_WIDTH 4
#endif
#ifndef SG_TUNED_UNROLL
#define SG_TUNED_UNROLL 1
#endif
#ifndef SG_TUNED_ASYNC_DEPTH
#define SG_TUNED_ASYNC_DEPTH 2
#endif

#include "csrc/backends/hip/gfx942/primitives.hpp"

namespace sg { namespace gfx942 {

namespace prim = ::sg::gfx942::primitives;

// ═══════════════════════════════════════════════════════════════════════
//  SuperGrok v2 — gfx942 functional port (CSA/HCA meta-model).
//
//  Per parameter p (with gradient g of length N and sharpness s):
//      1. sort_idx  = argsort(|g|, ascending)
//      2. x_sorted  = (input_proj_W @ [g, s].T + b)[sort_idx]   // [N, d_model]
//      3. csa_ctx   = unsort( csa_hca_attention(x_sorted, mode=csa, ...) )
//      4. hca_ctx   = unsort( csa_hca_attention(x_sorted, mode=hca, ...) )
//      5. ctx       = csa_ctx + hca_ctx                         // [N, d_model]
//      6. peer_out  = peer_route(ctx, query_W, prod_keys_{A,B}, expert_{W1,W2})
//      7. gru_new   = gru_step(peer_out, gru_state, gru_weights)
//      8. smart_g   = g + rescale * gru_new[:, 0]
//      9. mu        = alpha_mu * mu_prev + (1 - alpha_mu) * g
//     10. eff_grad  = smart_g + lamb_eff * mu
//     11. AdamW(eff_grad)
//
//  Steps 6-11 are UNCHANGED from the Mamba-3 era (spec §3b). Only the sequence
//  mixer (steps 3-5) changed from a bidirectional scan to CSA/HCA attention.
// ═══════════════════════════════════════════════════════════════════════


// ─── (helper) DeepSeek-V4 CSA / HCA hybrid attention, one mode.
//
// Replaces the former `mamba3_scan` helper. Mirrors the Python reference
// (grokking_optimizers/optimizers/supergrok2.py :: HybridCompressedAttention).
//
// Inputs:
//   x          [N, d_model]   FP32 (|g|-sorted)
//   q_W,k_W,v_W,out_W  [d_model, d_model]  (nn.Linear weights, applied as x@W.t())
//   compress_w [csa_window]   learned KV-pool weights (CSA only; ignored for HCA)
//   idx_DQ     [d_model, indexer_rank]  lightning-indexer low-rank query (CSA)
//   idx_UQ     [indexer_rank, d_model]  (CSA)
//   idx_K      [d_model, indexer_rank]  indexer key proj (CSA)
//   mode_csa   true → CSA (compress=csa_compress, top-k, +window);
//              false → HCA (stride-128 mean pool, dense over all + window)
//
// Output:
//   ctx        [N, d_model]   attention context (post out_W projection)
//
// CSA: strided weighted KV pool (window=csa_window, stride=csa_compress) →
//      lightning indexer (low-rank q vs compressed indexer keys) → top-k
//      (csa_topk, clamped to Nc) selection → causal sliding window → multi-head
//      joint softmax(QKᵀ/√head_dim)·V over (selected compressed ∪ window).
// HCA: stride-hca_compress mean pool → dense attention over all compressed
//      entries plus the causal sliding window.
static torch::Tensor csa_hca_attention(
    const torch::Tensor& x_in,
    const torch::Tensor& q_W,
    const torch::Tensor& k_W,
    const torch::Tensor& v_W,
    const torch::Tensor& out_W,
    const torch::Tensor& compress_w,
    const torch::Tensor& idx_DQ,
    const torch::Tensor& idx_UQ,
    const torch::Tensor& idx_K,
    bool mode_csa,
    int64_t num_heads,
    int64_t csa_compress,
    int64_t csa_window,
    int64_t csa_topk,
    int64_t hca_compress
) {
    using namespace torch::indexing;
    auto x = x_in.to(torch::kFloat32);
    const auto N = x.size(0);
    const auto d = x.size(1);

    auto opts_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(x.device());

    if (N == 0) {
        return torch::zeros({0, d}, opts_f32);
    }

    const auto head_dim = d / num_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const double sqrt_d = std::sqrt(static_cast<double>(d));

    // Per-token projections (nn.Linear: x @ W.t()).
    auto q     = torch::mm(x, q_W.t());   // [N, d]
    auto k_tok = torch::mm(x, k_W.t());   // [N, d]
    auto v_tok = torch::mm(x, v_W.t());   // [N, d]

    if (mode_csa) {
        const int64_t stride = csa_compress;
        const int64_t win    = csa_window;
        const int64_t nc     = (N + stride - 1) / stride;     // ceil(N/stride)

        // ── Strided weighted pooling of K/V into compressed entries ──
        // Compressed entry j pools x[j*stride : j*stride+win].
        auto pool_w = torch::softmax(compress_w, /*dim=*/0);  // [win]
        auto starts = torch::arange(nc, opts_f32.dtype(torch::kLong)) * stride;  // [Nc]
        auto offs   = torch::arange(win, opts_f32.dtype(torch::kLong));          // [win]
        auto gather = starts.unsqueeze(1) + offs.unsqueeze(0);                   // [Nc, win]
        auto valid  = gather < N;                                                // [Nc, win]
        auto gather_c = gather.clamp_max(N - 1);
        auto w_eff = pool_w.unsqueeze(0) * valid.to(torch::kFloat32);            // [Nc, win]
        w_eff = w_eff / w_eff.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(1e-12); // [Nc, win]

        // Pool compressed K/V: gather [Nc, win, d] weighted-sum over win.
        auto c_k = (k_tok.index({gather_c}) * w_eff.unsqueeze(-1)).sum(/*dim=*/1);  // [Nc, d]
        auto c_v = (v_tok.index({gather_c}) * w_eff.unsqueeze(-1)).sum(/*dim=*/1);  // [Nc, d]

        // ── Lightning indexer top-k selection ──
        auto qI     = torch::mm(torch::mm(x, idx_DQ), idx_UQ);        // [N, d]
        auto kI_tok = torch::mm(torch::mm(x, idx_K), idx_UQ);         // [N, d]
        auto c_kI   = (kI_tok.index({gather_c}) * w_eff.unsqueeze(-1)).sum(/*dim=*/1);  // [Nc, d]
        auto idx_scores = torch::mm(qI, c_kI.t()) / sqrt_d;           // [N, Nc]
        const int64_t topk = std::min<int64_t>(csa_topk, nc);
        auto sel = std::get<1>(idx_scores.topk(topk, /*dim=*/-1));    // [N, topk]

        // Gather selected compressed K/V per query → [N, topk, d].
        auto sel_k = c_k.index({sel});                               // [N, topk, d]
        auto sel_v = c_v.index({sel});                               // [N, topk, d]
        auto sel_kh = sel_k.reshape({N, topk, num_heads, head_dim});
        auto sel_vh = sel_v.reshape({N, topk, num_heads, head_dim});
        auto qh = q.reshape({N, num_heads, head_dim});               // [N, H, hd]
        // comp_scores[n,h,k] = sum_d qh[n,h,d] * sel_kh[n,k,h,d]
        auto comp_scores = torch::einsum("nhd,nkhd->nhk", {qh, sel_kh}) * scale;  // [N,H,topk]

        // ── Causal sliding window over raw tokens ──
        const int64_t wsz = std::min<int64_t>(win, N);
        auto woffs = torch::arange(wsz, opts_f32.dtype(torch::kLong));           // [wsz]
        auto qpos  = torch::arange(N, opts_f32.dtype(torch::kLong)).unsqueeze(1);// [N,1]
        auto win_idx = qpos - woffs.unsqueeze(0);                                // [N, wsz]
        auto win_valid = win_idx >= 0;                                           // [N, wsz]
        auto win_idx_c = win_idx.clamp_min(0);
        auto win_k = k_tok.index({win_idx_c}).reshape({N, wsz, num_heads, head_dim});
        auto win_v = v_tok.index({win_idx_c}).reshape({N, wsz, num_heads, head_dim});
        auto win_scores = torch::einsum("nhd,nwhd->nhw", {qh, win_k}) * scale;   // [N,H,wsz]
        win_scores = win_scores.masked_fill(
            win_valid.logical_not().unsqueeze(1),
            -std::numeric_limits<float>::infinity());

        // ── Joint softmax over (selected compressed ∪ window) ──
        auto all_scores = torch::cat({comp_scores, win_scores}, /*dim=*/-1);     // [N,H,topk+wsz]
        auto attn = torch::softmax(all_scores, /*dim=*/-1);
        auto attn_c = attn.index({Slice(), Slice(), Slice(0, topk)});            // [N,H,topk]
        auto attn_w = attn.index({Slice(), Slice(), Slice(topk, None)});         // [N,H,wsz]
        auto ctx_h = torch::einsum("nhk,nkhd->nhd", {attn_c, sel_vh})
                   + torch::einsum("nhw,nwhd->nhd", {attn_w, win_v});            // [N,H,hd]
        auto ctx = ctx_h.reshape({N, d});
        return torch::mm(ctx, out_W.t());                                        // [N, d]
    } else {
        // ── HCA: stride-128 mean pool, dense attention over all entries ──
        const int64_t stride = hca_compress;
        const int64_t nh     = (N + stride - 1) / stride;
        auto starts = torch::arange(nh, opts_f32.dtype(torch::kLong)) * stride;  // [Nh]
        auto offs   = torch::arange(stride, opts_f32.dtype(torch::kLong));       // [stride]
        auto gather = starts.unsqueeze(1) + offs.unsqueeze(0);                   // [Nh, stride]
        auto valid  = gather < N;
        auto gather_c = gather.clamp_max(N - 1);
        auto w_eff = valid.to(torch::kFloat32);
        w_eff = w_eff / w_eff.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(1e-12); // [Nh, stride]
        auto c_k = (k_tok.index({gather_c}) * w_eff.unsqueeze(-1)).sum(/*dim=*/1);  // [Nh, d]
        auto c_v = (v_tok.index({gather_c}) * w_eff.unsqueeze(-1)).sum(/*dim=*/1);  // [Nh, d]

        // Split heads → [H, L, hd].
        auto split_heads = [&](const torch::Tensor& t) {
            const auto L = t.size(0);
            return t.reshape({L, num_heads, head_dim}).permute({1, 0, 2}).contiguous();
        };
        auto qh   = split_heads(q);      // [H, N, hd]
        auto c_kh = split_heads(c_k);    // [H, Nh, hd]
        auto c_vh = split_heads(c_v);    // [H, Nh, hd]
        // Dense scores over all compressed entries: [H, N, Nh].
        auto comp_scores = torch::einsum("hnd,hmd->hnm", {qh, c_kh}) * scale;

        // Causal sliding window (reuse csa_window size for local context).
        const int64_t win = std::min<int64_t>(csa_window, N);
        auto woffs = torch::arange(win, opts_f32.dtype(torch::kLong));
        auto qpos  = torch::arange(N, opts_f32.dtype(torch::kLong)).unsqueeze(1);
        auto win_idx = qpos - woffs.unsqueeze(0);                                // [N, win]
        auto win_valid = win_idx >= 0;
        auto win_idx_c = win_idx.clamp_min(0);
        // [H, N, win, hd]: gather along the seq dim of split-head k/v.
        auto kh_full = split_heads(k_tok);  // [H, N, hd]
        auto vh_full = split_heads(v_tok);  // [H, N, hd]
        auto win_kh = kh_full.index({Slice(), win_idx_c, Slice()});             // [H, N, win, hd]
        auto win_vh = vh_full.index({Slice(), win_idx_c, Slice()});             // [H, N, win, hd]
        auto win_scores = torch::einsum("hnd,hnwd->hnw", {qh, win_kh}) * scale;  // [H, N, win]
        win_scores = win_scores.masked_fill(
            win_valid.logical_not().unsqueeze(0),
            -std::numeric_limits<float>::infinity());

        auto all_scores = torch::cat({comp_scores, win_scores}, /*dim=*/-1);     // [H, N, Nh+win]
        auto attn = torch::softmax(all_scores, /*dim=*/-1);
        auto attn_c = attn.index({Slice(), Slice(), Slice(0, nh)});             // [H, N, Nh]
        auto attn_w = attn.index({Slice(), Slice(), Slice(nh, None)});          // [H, N, win]
        auto ctx_h = torch::einsum("hnm,hmd->hnd", {attn_c, c_vh})
                   + torch::einsum("hnw,hnwd->hnd", {attn_w, win_vh});          // [H, N, hd]
        auto ctx = ctx_h.permute({1, 0, 2}).reshape({N, d});
        return torch::mm(ctx, out_W.t());                                        // [N, d]
    }
}


// ─── (helper) PEER routing with soft expert MLP.  UNCHANGED.
//
// Inputs:
//   ctx          [N, d_model]  — combined CSA + HCA context (unsorted)
//   peer_query_Ws[num_heads, d_model, d_model]  — per-head query proj
//   prod_keys_A  [num_heads, num_keys, half_qd]  — product keys, half A
//   prod_keys_B  [num_heads, num_keys, half_qd]  — product keys, half B
//   expert_W1    [num_experts, expert_hidden]
//   expert_W2    [num_experts, expert_hidden]
//   topk         scalar
//
// Output:
//   peer_out     [N]  — per-element soft expert output
//   expert_use   [num_experts]  — count of activations (int32, for recycling)
static std::pair<torch::Tensor, torch::Tensor> peer_route(
    const torch::Tensor& ctx,
    const torch::Tensor& peer_query_Ws,
    const torch::Tensor& prod_keys_A,
    const torch::Tensor& prod_keys_B,
    const torch::Tensor& expert_W1,
    const torch::Tensor& expert_W2,
    int64_t num_experts,
    int64_t topk
) {
    using namespace torch::indexing;
    const auto N = ctx.size(0);
    const auto num_heads = peer_query_Ws.size(0);

    auto opts_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(ctx.device());

    auto peer_total = torch::zeros({N}, opts_f32);
    auto expert_use = torch::zeros({num_experts},
        torch::TensorOptions().dtype(torch::kInt32).device(ctx.device()));

    // num_keys = sqrt(num_experts)
    const auto num_keys = static_cast<int64_t>(std::sqrt(static_cast<double>(num_experts)));
    const auto half_qd = prod_keys_A.size(2);

    for (int64_t h = 0; h < num_heads; ++h) {
        // Per-head: project ctx through Wq, split into two halves.
        auto q = torch::mm(ctx, peer_query_Ws.index({h}).t());   // [N, d_qd]
        auto q_a = q.index({Slice(), Slice(0, half_qd)});         // [N, half_qd]
        auto q_b = q.index({Slice(), Slice(half_qd, 2 * half_qd)});

        auto keys_a = prod_keys_A.index({h});                     // [num_keys, half_qd]
        auto keys_b = prod_keys_B.index({h});

        // Scores: per-half dot products with keys.
        auto scores_a = torch::mm(q_a, keys_a.t());               // [N, num_keys]
        auto scores_b = torch::mm(q_b, keys_b.t());

        // Top-k along each half.
        auto top_a = scores_a.topk(topk, /*dim=*/1);
        auto top_b = scores_b.topk(topk, /*dim=*/1);

        auto top_a_vals = std::get<0>(top_a);                     // [N, topk]
        auto top_a_idx  = std::get<1>(top_a);                     // [N, topk]
        auto top_b_vals = std::get<0>(top_b);
        auto top_b_idx  = std::get<1>(top_b);

        // Outer product of the top-k from each half → topk*topk candidates.
        auto pair_scores = top_a_vals.unsqueeze(2) + top_b_vals.unsqueeze(1);  // [N, topk, topk]
        auto pair_idx_a = top_a_idx.unsqueeze(2).expand({N, topk, topk});
        auto pair_idx_b = top_b_idx.unsqueeze(1).expand({N, topk, topk});
        auto pair_expert = pair_idx_a * num_keys + pair_idx_b;     // [N, topk, topk]

        auto pair_flat   = pair_scores.reshape({N, topk * topk});
        auto pair_expert_flat = pair_expert.reshape({N, topk * topk});

        // Softmax over the topk*topk candidates per element.
        auto routing_w = torch::softmax(pair_flat, /*dim=*/1);

        // Expert evaluation: each candidate runs through its expert MLP.
        auto kk = topk * topk;
        for (int64_t cand = 0; cand < kk; ++cand) {
            auto eidx = pair_expert_flat.index({Slice(), cand});          // [N] int
            auto eW1  = expert_W1.index_select(0, eidx);                  // [N, expert_hidden]
            auto eW2  = expert_W2.index_select(0, eidx);                  // [N, expert_hidden]

            // Simple expert MLP: input is the routing score itself (scalar per element).
            auto inp = top_a_vals.index({Slice(), cand / topk}) +
                       top_b_vals.index({Slice(), cand % topk});           // [N]
            auto hidden = (eW1 * inp.unsqueeze(1)).relu();                 // [N, expert_hidden]
            auto out    = (eW2 * hidden).sum(1);                           // [N]

            peer_total.add_(routing_w.index({Slice(), cand}) * out);

            // Track expert activations for recycling. scatter_add on int32.
            auto ones = torch::ones_like(eidx, torch::TensorOptions()
                .dtype(torch::kInt32).device(ctx.device()));
            expert_use.scatter_add_(0, eidx, ones);
        }
    }

    return std::make_pair(peer_total / num_heads, expert_use);
}


// ─── (helper) per-element GRU.  UNCHANGED.
//
// h_new = (1 - z) * h_old + z * tanh(Wh @ [x; r * h_old] + bh)
// z = sigmoid(Wz @ [x; h_old] + bz)
// r = sigmoid(Wr @ [x; h_old] + br)
static torch::Tensor gru_step(
    const torch::Tensor& x,         // [N, in_dim]
    const torch::Tensor& h_old,     // [N, hidden]
    const torch::Tensor& Wz,        // [hidden, in_dim + hidden]
    const torch::Tensor& bz,
    const torch::Tensor& Wr,
    const torch::Tensor& br,
    const torch::Tensor& Wh,
    const torch::Tensor& bh
) {
    auto xh   = torch::cat({x, h_old}, /*dim=*/1);
    auto z    = torch::sigmoid(torch::addmm(bz.unsqueeze(0), xh, Wz.t()));
    auto r    = torch::sigmoid(torch::addmm(br.unsqueeze(0), xh, Wr.t()));
    auto xrh  = torch::cat({x, r * h_old}, /*dim=*/1);
    auto h_tilde = torch::tanh(torch::addmm(bh.unsqueeze(0), xrh, Wh.t()));
    return (1.0f - z) * h_old + z * h_tilde;
}


// ═══════════════════════════════════════════════════════════════════════
//  Per-parameter SG2 step (CSA/HCA).
//
//  Used by `launch_csa_hca_step` (once per parameter) and as the inner body of
//  `launch_csa_hca_batched_step` (host-side for-loop over the parameter list).
// ═══════════════════════════════════════════════════════════════════════
static void sg2_step_one_param(
    torch::Tensor& param,
    const torch::Tensor& grad,
    const torch::Tensor& sharpness,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& mu,
    torch::Tensor& gru_state,
    // input projection
    const torch::Tensor& input_proj_W,
    const torch::Tensor& input_proj_b,
    // CSA layer weights (produce csa_ctx)
    const torch::Tensor& csa_q_W,
    const torch::Tensor& csa_k_W,
    const torch::Tensor& csa_v_W,
    const torch::Tensor& csa_compress_w,
    const torch::Tensor& csa_idx_DQ,
    const torch::Tensor& csa_idx_UQ,
    const torch::Tensor& csa_idx_K,
    const torch::Tensor& csa_out_W,
    // HCA layer weights (produce hca_ctx)
    const torch::Tensor& hca_q_W,
    const torch::Tensor& hca_k_W,
    const torch::Tensor& hca_v_W,
    const torch::Tensor& hca_out_W,
    // GRU
    const torch::Tensor& gru_Wz, const torch::Tensor& gru_bz,
    const torch::Tensor& gru_Wr, const torch::Tensor& gru_br,
    const torch::Tensor& gru_Wh, const torch::Tensor& gru_bh,
    // PEER routing
    const torch::Tensor& peer_query_Ws,
    const torch::Tensor& prod_keys_A,
    const torch::Tensor& prod_keys_B,
    const torch::Tensor& expert_W1,
    const torch::Tensor& expert_W2,
    // Adam hyperparams
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    // attention dims
    int64_t num_heads,
    int64_t csa_compress, int64_t csa_window, int64_t csa_topk,
    int64_t hca_compress, int64_t indexer_rank,
    int64_t num_experts, int64_t topk,
    torch::Tensor& expert_counts
) {
    using namespace torch::indexing;
    (void)indexer_rank;  // implied by idx_* tensor shapes

    const auto N = param.numel();
    if (N == 0) return;

    auto p_flat = param.reshape({N}).to(torch::kFloat32);
    auto g_flat = grad.reshape({N}).to(torch::kFloat32);
    auto s_flat = sharpness.reshape({N}).to(torch::kFloat32);

    // (1) sort by |g| ascending
    auto abs_g = g_flat.abs();
    auto sort_pair = abs_g.sort(/*dim=*/0, /*descending=*/false);
    auto sort_idx = std::get<1>(sort_pair);

    // unsort = inverse permutation
    auto unsort_idx = torch::empty_like(sort_idx);
    auto arange_N = torch::arange(N, sort_idx.options());
    unsort_idx.scatter_(0, sort_idx, arange_N);

    auto g_sorted = g_flat.index_select(0, sort_idx);
    auto s_sorted = s_flat.index_select(0, sort_idx);

    // (2) input projection: [g, s] (rescaled) → x_proj [N, d_model]
    auto inp = torch::stack({g_sorted * rescale, s_sorted * rescale}, /*dim=*/1);
    auto x_proj = torch::addmm(input_proj_b.unsqueeze(0), inp, input_proj_W.t());

    // (3) CSA attention → fine-grained / local context (was mamba_fwd).
    auto csa_sorted = csa_hca_attention(
        x_proj, csa_q_W, csa_k_W, csa_v_W, csa_out_W,
        csa_compress_w, csa_idx_DQ, csa_idx_UQ, csa_idx_K,
        /*mode_csa=*/true, num_heads,
        csa_compress, csa_window, csa_topk, hca_compress);

    // (4) HCA attention → global coarse context (was mamba_bwd). The HCA layer
    //     has no indexer / compress_w; pass empty placeholders.
    auto empty = torch::Tensor{};
    auto hca_sorted = csa_hca_attention(
        x_proj, hca_q_W, hca_k_W, hca_v_W, hca_out_W,
        /*compress_w=*/empty, /*idx_DQ=*/empty, /*idx_UQ=*/empty, /*idx_K=*/empty,
        /*mode_csa=*/false, num_heads,
        csa_compress, csa_window, csa_topk, hca_compress);

    // (5) unsort both, combine.
    auto csa_ctx = csa_sorted.index_select(0, unsort_idx);          // [N, d_model]
    auto hca_ctx = hca_sorted.index_select(0, unsort_idx);          // [N, d_model]
    auto ctx = csa_ctx + hca_ctx;                                   // [N, d_model]

    // (6) PEER routing.  UNCHANGED.
    auto peer_out_pair = peer_route(
        ctx, peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_W2, num_experts, topk);
    auto peer_out = peer_out_pair.first;                            // [N]
    expert_counts.add_(peer_out_pair.second);

    // (7) per-element GRU (treat gru_state as [gru_hidden], broadcast across N).
    //     UNCHANGED.
    const auto gru_hidden = gru_state.size(0);
    auto gru_state_2d = gru_state.unsqueeze(0).expand({N, gru_hidden}).contiguous();
    auto peer_inp = peer_out.unsqueeze(1);                           // [N, 1]

    auto gru_new = gru_step(
        peer_inp, gru_state_2d,
        gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh);            // [N, H]

    // Persist state: mean across the N dim (shared GRU state per parameter).
    gru_state.copy_(gru_new.mean(0));

    // (8) smart_grad: g + rescale * gru[:, 0].  UNCHANGED.
    auto smart_g = g_flat + rescale * gru_new.index({Slice(), 0});

    // (9) mu update and effective gradient.  UNCHANGED.
    mu.reshape({N}).mul_(alpha_mu).add_(g_flat, 1.0f - alpha_mu);
    auto eff_grad = smart_g + lamb_eff * mu.reshape({N});

    // (10) AdamW step.  UNCHANGED.
    auto ea = exp_avg.reshape({N});
    auto easq = exp_avg_sq.reshape({N});
    ea.mul_(beta1).add_(eff_grad, 1.0f - beta1);
    easq.mul_(beta2).addcmul_(eff_grad, eff_grad, 1.0f - beta2);

    auto m_hat = ea / bc1;
    auto v_hat = easq / bc2;
    auto denom = v_hat.sqrt().add_(eps);
    auto update = m_hat / denom;
    p_flat.mul_(1.0f - lr * wd_eff).add_(update, -lr);

    // Cast back to original param dtype.
    param.reshape({N}).copy_(p_flat.to(param.dtype()));
}


// ═══════════════════════════════════════════════════════════════════════
//  Public launchers — bindings-expected entry points.
//
//  These match the CSA/HCA signatures in spec §6/§7 and live in
//  `sg::gfx942::*` (matched against bindings via the DISPATCH_SG2 macro).
// ═══════════════════════════════════════════════════════════════════════

// Spec §7 — locked CSA/HCA single-tensor launcher parameter list.
void launch_csa_hca_step(
    torch::Tensor param, torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor exp_avg, torch::Tensor exp_avg_sq, torch::Tensor mu,
    torch::Tensor gru_state,
    // --- shared input projection ---
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    // --- CSA layer (produces csa_ctx) ---
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    // --- HCA layer (produces hca_ctx) ---
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    // --- GRU (carried across steps) ---
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr,
    torch::Tensor gru_br, torch::Tensor gru_Wh, torch::Tensor gru_bh,
    // --- PEER routing + experts ---
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    // --- scalars ---
    float rescale, float alpha_mu, float lamb_eff,
    float beta1, float beta2, float lr, float wd_eff, float eps,
    float bc1, float bc2,
    int d_model, int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor expert_counts
) {
    (void)expert_b1; (void)expert_b2;   // expert MLP biases unused on this path
    (void)d_model; (void)gru_hidden; (void)pk_dim; (void)expert_hidden;
    sg2_step_one_param(
        param, grad, sharpness, exp_avg, exp_avg_sq, mu, gru_state,
        input_proj_W, input_proj_b,
        csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
        csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
        hca_q_W, hca_k_W, hca_v_W, hca_out_W,
        gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
        peer_query_Ws, prod_keys_A, prod_keys_B,
        expert_W1, expert_W2,
        rescale, alpha_mu, lamb_eff,
        beta1, beta2, lr, wd_eff, eps, bc1, bc2,
        num_heads, csa_compress, csa_window, csa_topk,
        hca_compress, indexer_rank,
        num_experts, /*topk=*/4, expert_counts);
}


// Spec §7 — batched variant: vectors for the per-tensor plumbing + scalars,
// shared meta-weights passed once (mirrors the single-tensor list, minus the
// mamba states).
void launch_csa_hca_batched_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    std::vector<torch::Tensor> exp_avgs,
    std::vector<torch::Tensor> exp_avg_sqs,
    std::vector<torch::Tensor> mus,
    std::vector<torch::Tensor> gru_states,
    // --- shared input projection ---
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    // --- CSA layer ---
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    // --- HCA layer ---
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    // --- GRU ---
    torch::Tensor gru_Wz, torch::Tensor gru_bz, torch::Tensor gru_Wr,
    torch::Tensor gru_br, torch::Tensor gru_Wh, torch::Tensor gru_bh,
    // --- PEER routing + experts ---
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_b1,
    torch::Tensor expert_W2, torch::Tensor expert_b2,
    // --- per-tensor scalars ---
    std::vector<float> alpha_mus, std::vector<float> lamb_effs,
    std::vector<float> beta1s,
    std::vector<float> bc1s, std::vector<float> bc2s,
    // --- shared scalars ---
    float rescale, float beta2, float lr, float wd_eff, float eps,
    int d_model, int gru_hidden, int num_heads, int pk_dim,
    int expert_hidden, int num_experts,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor expert_counts
) {
    (void)expert_b1; (void)expert_b2;
    const size_t n_params = params.size();
    TORCH_CHECK(grads.size()           == n_params, "grads size mismatch");
    TORCH_CHECK(sharpness_list.size()  == n_params, "sharpness size mismatch");
    TORCH_CHECK(exp_avgs.size()        == n_params, "exp_avgs size mismatch");
    TORCH_CHECK(exp_avg_sqs.size()     == n_params, "exp_avg_sqs size mismatch");
    TORCH_CHECK(mus.size()             == n_params, "mus size mismatch");
    TORCH_CHECK(gru_states.size()      == n_params, "gru_states size mismatch");
    TORCH_CHECK(alpha_mus.size()       == n_params, "alpha_mus size mismatch");

    for (size_t i = 0; i < n_params; ++i) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        sg2_step_one_param(
            params[i], grads[i], sharpness_list[i],
            exp_avgs[i], exp_avg_sqs[i], mus[i], gru_states[i],
            input_proj_W, input_proj_b,
            csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
            csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
            hca_q_W, hca_k_W, hca_v_W, hca_out_W,
            gru_Wz, gru_bz, gru_Wr, gru_br, gru_Wh, gru_bh,
            peer_query_Ws, prod_keys_A, prod_keys_B,
            expert_W1, expert_W2,
            rescale, alpha_mus[i], lamb_effs[i],
            beta1s[i], beta2, lr, wd_eff, eps, bc1s[i], bc2s[i],
            num_heads, csa_compress, csa_window, csa_topk,
            hca_compress, indexer_rank,
            num_experts, /*topk=*/4, expert_counts);
    }
    (void)d_model; (void)gru_hidden; (void)pk_dim; (void)expert_hidden;
}


// ═══════════════════════════════════════════════════════════════════════
//  Bilevel forward-save + backward — REAL hand-written ATen adjoint.
//
//  Implemented per the AMD-native stance: the math (a full reverse-mode VJP
//  through the CSA/HCA meta-net) lives in the vendor-neutral header
//  csrc/algorithms/supergrok2_bilevel_adjoint.h and is shared bit-for-bit with
//  sm_90. This file is ATen-based (host-compiler TU), matching the existing
//  gfx942 SG2 style; Stage 5's AMD-native rewrite will lower it to raw HIP.
//  NO throw remains. Signatures are locked to bindings.cpp::DECLARE_SG2(gfx942).
// ═══════════════════════════════════════════════════════════════════════

namespace sg2adj = ::sg::algorithms::sg2_bilevel;

void launch_csa_hca_bilevel_fwd_save(
    torch::Tensor grad, torch::Tensor sharpness,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    int d_model, int num_heads,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor csa_ctx_out, torch::Tensor hca_ctx_out,
    torch::Tensor x_sorted, torch::Tensor sort_indices,
    torch::Tensor csa_saved_denom, torch::Tensor csa_saved_sel_idx,
    torch::Tensor csa_saved_probs,
    torch::Tensor hca_saved_denom, torch::Tensor hca_saved_probs,
    int checkpoint_interval)
{
    if (grad.numel() == 0) return;
    (void)checkpoint_interval;
    auto fopt = grad.options().dtype(torch::kFloat32);
    auto h0 = torch::zeros({4}, fopt);
    auto S = sg2adj::bilevel_forward_save(
        grad, sharpness, h0, input_proj_W, input_proj_b,
        csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
        csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
        hca_q_W, hca_k_W, hca_v_W, hca_out_W,
        torch::zeros({4, 2 + 2 * d_model}, fopt), torch::zeros({4}, fopt),
        torch::zeros({4, 2 + 2 * d_model}, fopt), torch::zeros({4}, fopt),
        torch::zeros({4, 2 + 2 * d_model}, fopt), torch::zeros({4}, fopt),
        d_model, num_heads, csa_compress, csa_window, csa_topk,
        hca_compress, indexer_rank);
    if (csa_ctx_out.defined() && csa_ctx_out.numel() > 0) csa_ctx_out.copy_(S.csa_ctx);
    if (hca_ctx_out.defined() && hca_ctx_out.numel() > 0) hca_ctx_out.copy_(S.hca_ctx);
    if (x_sorted.defined() && x_sorted.numel() > 0) x_sorted.copy_(S.x_sorted);
    if (sort_indices.defined() && sort_indices.numel() > 0)
        sort_indices.copy_(S.sort_idx.to(sort_indices.dtype()));
    if (csa_saved_denom.defined() && csa_saved_denom.numel() > 0)
        csa_saved_denom.copy_(S.csa_denom);
    if (csa_saved_sel_idx.defined() && csa_saved_sel_idx.numel() > 0)
        csa_saved_sel_idx.copy_(S.csa_sel_idx.to(csa_saved_sel_idx.dtype()));
    if (csa_saved_probs.defined() && csa_saved_probs.numel() > 0)
        csa_saved_probs.copy_(S.csa_probs.reshape(csa_saved_probs.sizes()));
    if (hca_saved_denom.defined() && hca_saved_denom.numel() > 0)
        hca_saved_denom.copy_(S.hca_denom);
    if (hca_saved_probs.defined() && hca_saved_probs.numel() > 0)
        hca_saved_probs.copy_(S.hca_probs.reshape(hca_saved_probs.sizes()));
}

void launch_csa_hca_bilevel_fwd_save_batched(
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> sharpness_list,
    torch::Tensor input_proj_W, torch::Tensor input_proj_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    int d_model, int num_heads,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    torch::Tensor csa_ctx_out_packed, torch::Tensor hca_ctx_out_packed,
    torch::Tensor x_sorted_packed, torch::Tensor offsets_t,
    torch::Tensor sort_indices_packed,
    torch::Tensor csa_saved_denom_packed, torch::Tensor csa_saved_sel_idx_packed,
    torch::Tensor csa_saved_probs_packed,
    torch::Tensor hca_saved_denom_packed, torch::Tensor hca_saved_probs_packed,
    int checkpoint_interval)
{
    if (grads.empty()) return;
    (void)checkpoint_interval;
    auto offs = offsets_t.to(torch::kCPU).to(torch::kLong);
    auto oacc = offs.accessor<int64_t, 1>();
    const int P = (int)grads.size();
    for (int p = 0; p < P; ++p) {
        auto& g = grads[p];
        if (!g.defined() || g.numel() == 0) continue;
        const int64_t start = oacc[p], end = oacc[p + 1], n = end - start;
        if (n <= 0) continue;
        auto fopt = g.options().dtype(torch::kFloat32);
        auto S = sg2adj::bilevel_forward_save(
            g, sharpness_list[p], torch::zeros({4}, fopt), input_proj_W, input_proj_b,
            csa_q_W, csa_k_W, csa_v_W, csa_compress_w,
            csa_idx_DQ, csa_idx_UQ, csa_idx_K, csa_out_W,
            hca_q_W, hca_k_W, hca_v_W, hca_out_W,
            torch::zeros({4, 2 + 2 * d_model}, fopt), torch::zeros({4}, fopt),
            torch::zeros({4, 2 + 2 * d_model}, fopt), torch::zeros({4}, fopt),
            torch::zeros({4, 2 + 2 * d_model}, fopt), torch::zeros({4}, fopt),
            d_model, num_heads, csa_compress, csa_window, csa_topk,
            hca_compress, indexer_rank);
        if (csa_ctx_out_packed.defined() && csa_ctx_out_packed.numel() > 0)
            csa_ctx_out_packed.narrow(0, start, n).copy_(S.csa_ctx);
        if (hca_ctx_out_packed.defined() && hca_ctx_out_packed.numel() > 0)
            hca_ctx_out_packed.narrow(0, start, n).copy_(S.hca_ctx);
        if (x_sorted_packed.defined() && x_sorted_packed.numel() > 0)
            x_sorted_packed.narrow(0, start, n).copy_(S.x_sorted);
        if (sort_indices_packed.defined() && sort_indices_packed.numel() > 0)
            sort_indices_packed.narrow(0, start, n).copy_(
                S.sort_idx.to(sort_indices_packed.dtype()));
    }
    (void)csa_saved_denom_packed; (void)csa_saved_sel_idx_packed;
    (void)csa_saved_probs_packed; (void)hca_saved_denom_packed;
    (void)hca_saved_probs_packed;
}

void launch_csa_hca_backward(
    torch::Tensor d_smart_grad,
    torch::Tensor grad, torch::Tensor sharpness, float rescale,
    torch::Tensor sort_indices, torch::Tensor x_sorted,
    torch::Tensor csa_ctx, torch::Tensor hca_ctx,
    torch::Tensor csa_saved_denom, torch::Tensor csa_saved_sel_idx,
    torch::Tensor csa_saved_probs,
    torch::Tensor hca_saved_denom, torch::Tensor hca_saved_probs,
    torch::Tensor gru_input, torch::Tensor gru_h_old,
    torch::Tensor gru_z_gate, torch::Tensor gru_r_gate, torch::Tensor gru_h_tilde,
    torch::Tensor peer_input, torch::Tensor expert_indices,
    torch::Tensor routing_weights, torch::Tensor saved_z_hidden,
    torch::Tensor saved_scores_a, torch::Tensor saved_scores_b,
    torch::Tensor saved_top_a_idx, torch::Tensor saved_top_b_idx,
    torch::Tensor saved_soft_a, torch::Tensor saved_soft_b,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    torch::Tensor gru_Wz, torch::Tensor gru_Wr, torch::Tensor gru_Wh,
    torch::Tensor peer_query_Ws, torch::Tensor prod_keys_A, torch::Tensor prod_keys_B,
    torch::Tensor expert_W1, torch::Tensor expert_W2,
    torch::Tensor expert_b1_in, torch::Tensor expert_b2_in,
    torch::Tensor input_proj_W,
    torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W, torch::Tensor d_csa_v_W,
    torch::Tensor d_csa_compress_w,
    torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ, torch::Tensor d_csa_idx_K,
    torch::Tensor d_csa_out_W,
    torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W, torch::Tensor d_hca_v_W,
    torch::Tensor d_hca_out_W,
    torch::Tensor d_gru_Wz, torch::Tensor d_gru_bz,
    torch::Tensor d_gru_Wr, torch::Tensor d_gru_br,
    torch::Tensor d_gru_Wh, torch::Tensor d_gru_bh,
    torch::Tensor d_peer_query_Ws,
    torch::Tensor d_prod_keys_A, torch::Tensor d_prod_keys_B,
    torch::Tensor d_expert_W1, torch::Tensor d_expert_b1,
    torch::Tensor d_expert_W2, torch::Tensor d_expert_b2,
    torch::Tensor d_input_proj_W, torch::Tensor d_input_proj_b,
    int d_model, int gru_hidden, int gru_input_dim,
    int num_heads, int topk, int pk_dim,
    int expert_hidden, int peer_input_dim, int num_experts,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    int checkpoint_interval)
{
    if (d_smart_grad.numel() == 0) return;
    (void)checkpoint_interval; (void)num_experts; (void)peer_input_dim;
    (void)gru_input_dim; (void)csa_saved_denom; (void)csa_saved_probs;
    (void)hca_saved_denom; (void)hca_saved_probs; (void)expert_indices;
    (void)routing_weights; (void)saved_z_hidden; (void)saved_scores_a;
    (void)saved_scores_b; (void)saved_top_a_idx; (void)saved_top_b_idx;
    (void)saved_soft_a; (void)saved_soft_b;

    auto fopt = x_sorted.options().dtype(torch::kFloat32);
    sg2adj::SavedActs S;
    S.g_col = grad.reshape({-1}).to(torch::kFloat32);
    S.s_col = sharpness.reshape({-1}).to(torch::kFloat32);
    S.x_sorted = x_sorted.to(torch::kFloat32);
    S.sort_idx = sort_indices.to(torch::kLong);
    S.unsort_idx = S.sort_idx.argsort();
    S.csa_ctx = csa_ctx.to(torch::kFloat32);
    S.hca_ctx = hca_ctx.to(torch::kFloat32);
    S.csa_sel_idx = (csa_saved_sel_idx.defined() && csa_saved_sel_idx.numel() > 0)
        ? csa_saved_sel_idx.to(torch::kLong) : torch::Tensor{};
    S.peer_input = peer_input.to(torch::kFloat32);
    S.gru_input  = gru_input.to(torch::kFloat32);
    S.gru_h_old  = gru_h_old.to(torch::kFloat32);

    auto xh = torch::cat({S.gru_input, S.gru_h_old}, -1);
    S.gru_z = (gru_z_gate.defined() && gru_z_gate.numel() > 0)
        ? gru_z_gate.to(torch::kFloat32)
        : torch::sigmoid(sg2adj::linear_fwd(xh, gru_Wz));
    S.gru_r = (gru_r_gate.defined() && gru_r_gate.numel() > 0)
        ? gru_r_gate.to(torch::kFloat32)
        : torch::sigmoid(sg2adj::linear_fwd(xh, gru_Wr));
    auto xrh = torch::cat({S.gru_input, S.gru_r * S.gru_h_old}, -1);
    S.gru_h_tilde = (gru_h_tilde.defined() && gru_h_tilde.numel() > 0)
        ? gru_h_tilde.to(torch::kFloat32)
        : torch::tanh(sg2adj::linear_fwd(xrh, gru_Wh));

    std::vector<torch::Tensor> peer_Wq, prod_A, prod_B, dpeer_Wq, dprod_A, dprod_B;
    const int64_t nph = peer_query_Ws.size(0);
    for (int64_t h = 0; h < nph; ++h) {
        peer_Wq.push_back(peer_query_Ws.index({h}).to(torch::kFloat32));
        prod_A.push_back(prod_keys_A.index({h}).to(torch::kFloat32));
        prod_B.push_back(prod_keys_B.index({h}).to(torch::kFloat32));
        dpeer_Wq.push_back(torch::zeros_like(peer_Wq.back()));
        dprod_A.push_back(torch::zeros_like(prod_A.back()));
        dprod_B.push_back(torch::zeros_like(prod_B.back()));
    }

    sg2adj::bilevel_backward_driver(
        d_smart_grad, rescale, S,
        input_proj_W.to(torch::kFloat32),
        csa_q_W.to(torch::kFloat32), csa_k_W.to(torch::kFloat32),
        csa_v_W.to(torch::kFloat32), csa_compress_w.to(torch::kFloat32),
        csa_idx_DQ.to(torch::kFloat32), csa_idx_UQ.to(torch::kFloat32),
        csa_idx_K.to(torch::kFloat32), csa_out_W.to(torch::kFloat32),
        hca_q_W.to(torch::kFloat32), hca_k_W.to(torch::kFloat32),
        hca_v_W.to(torch::kFloat32), hca_out_W.to(torch::kFloat32),
        gru_Wz.to(torch::kFloat32), gru_Wr.to(torch::kFloat32),
        gru_Wh.to(torch::kFloat32),
        peer_Wq, prod_A, prod_B,
        expert_W1.to(torch::kFloat32),
        (expert_b1_in.defined() && expert_b1_in.numel() > 0)
            ? expert_b1_in.to(torch::kFloat32)
            : torch::zeros({num_experts, expert_hidden}, fopt),
        expert_W2.to(torch::kFloat32),
        (expert_b2_in.defined() && expert_b2_in.numel() > 0)
            ? expert_b2_in.to(torch::kFloat32)
            : torch::zeros({num_experts, 1}, fopt),
        d_model, num_heads, gru_hidden, pk_dim, topk, expert_hidden,
        csa_compress, csa_window, csa_topk, hca_compress, indexer_rank,
        d_input_proj_W, d_input_proj_b,
        d_csa_q_W, d_csa_k_W, d_csa_v_W, d_csa_compress_w,
        d_csa_idx_DQ, d_csa_idx_UQ, d_csa_idx_K, d_csa_out_W,
        d_hca_q_W, d_hca_k_W, d_hca_v_W, d_hca_out_W,
        d_gru_Wz, d_gru_bz, d_gru_Wr, d_gru_br, d_gru_Wh, d_gru_bh,
        dpeer_Wq, dprod_A, dprod_B,
        d_expert_W1, d_expert_b1, d_expert_W2, d_expert_b2);

    for (int64_t h = 0; h < nph; ++h) {
        if (d_peer_query_Ws.defined() && d_peer_query_Ws.numel() > 0)
            d_peer_query_Ws.index({h}).add_(dpeer_Wq[h]);
        if (d_prod_keys_A.defined() && d_prod_keys_A.numel() > 0)
            d_prod_keys_A.index({h}).add_(dprod_A[h]);
        if (d_prod_keys_B.defined() && d_prod_keys_B.numel() > 0)
            d_prod_keys_B.index({h}).add_(dprod_B[h]);
    }
}

void launch_csa_hca_backward_batched(
    torch::Tensor d_csa_ctx_packed, torch::Tensor d_hca_ctx_packed,
    torch::Tensor x_sorted_packed,
    torch::Tensor csa_saved_denom_packed, torch::Tensor csa_saved_sel_idx_packed,
    torch::Tensor csa_saved_probs_packed,
    torch::Tensor hca_saved_denom_packed, torch::Tensor hca_saved_probs_packed,
    torch::Tensor offsets_t,
    torch::Tensor csa_q_W, torch::Tensor csa_k_W, torch::Tensor csa_v_W,
    torch::Tensor csa_compress_w,
    torch::Tensor csa_idx_DQ, torch::Tensor csa_idx_UQ, torch::Tensor csa_idx_K,
    torch::Tensor csa_out_W,
    torch::Tensor hca_q_W, torch::Tensor hca_k_W, torch::Tensor hca_v_W,
    torch::Tensor hca_out_W,
    torch::Tensor d_csa_q_W, torch::Tensor d_csa_k_W, torch::Tensor d_csa_v_W,
    torch::Tensor d_csa_compress_w,
    torch::Tensor d_csa_idx_DQ, torch::Tensor d_csa_idx_UQ, torch::Tensor d_csa_idx_K,
    torch::Tensor d_csa_out_W,
    torch::Tensor d_hca_q_W, torch::Tensor d_hca_k_W, torch::Tensor d_hca_v_W,
    torch::Tensor d_hca_out_W,
    torch::Tensor d_x_sorted_packed,
    int d_model, int num_heads, int num_params,
    int csa_compress, int csa_window, int csa_topk,
    int hca_compress, int indexer_rank,
    int checkpoint_interval)
{
    if (num_params == 0) return;
    (void)checkpoint_interval;
    (void)csa_saved_denom_packed; (void)csa_saved_probs_packed;
    (void)hca_saved_denom_packed; (void)hca_saved_probs_packed;
    (void)csa_saved_sel_idx_packed;
    auto offs = offsets_t.to(torch::kCPU).to(torch::kLong);
    auto oacc = offs.accessor<int64_t, 1>();
    for (int p = 0; p < num_params; ++p) {
        const int64_t start = oacc[p], end = oacc[p + 1], n = end - start;
        if (n <= 0) continue;
        auto x = x_sorted_packed.narrow(0, start, n).to(torch::kFloat32);
        auto d_csa = d_csa_ctx_packed.narrow(0, start, n).to(torch::kFloat32);
        auto d_hca = d_hca_ctx_packed.narrow(0, start, n).to(torch::kFloat32);
        auto d_x = torch::zeros_like(x);

        auto cf = sg2adj::csa_forward(
            x, csa_q_W.to(torch::kFloat32), csa_k_W.to(torch::kFloat32),
            csa_v_W.to(torch::kFloat32), csa_compress_w.to(torch::kFloat32),
            csa_idx_DQ.to(torch::kFloat32), csa_idx_UQ.to(torch::kFloat32),
            csa_idx_K.to(torch::kFloat32), num_heads,
            csa_compress, csa_window, csa_topk);
        d_csa_out_W.add_(torch::mm(d_csa.t(), cf.ctx));
        auto d_csa_pre = torch::mm(d_csa, csa_out_W.to(torch::kFloat32));
        sg2adj::csa_backward(
            x, cf, csa_q_W.to(torch::kFloat32), csa_k_W.to(torch::kFloat32),
            csa_v_W.to(torch::kFloat32), csa_compress_w.to(torch::kFloat32),
            csa_idx_DQ.to(torch::kFloat32), csa_idx_UQ.to(torch::kFloat32),
            csa_idx_K.to(torch::kFloat32), num_heads, d_csa_pre,
            d_csa_q_W, d_csa_k_W, d_csa_v_W, d_csa_compress_w,
            d_csa_idx_DQ, d_csa_idx_UQ, d_csa_idx_K, d_x);

        auto hf = sg2adj::hca_forward(
            x, hca_q_W.to(torch::kFloat32), hca_k_W.to(torch::kFloat32),
            hca_v_W.to(torch::kFloat32), num_heads, hca_compress, csa_window);
        d_hca_out_W.add_(torch::mm(d_hca.t(), hf.ctx));
        auto d_hca_pre = torch::mm(d_hca, hca_out_W.to(torch::kFloat32));
        sg2adj::hca_backward(
            x, hf, hca_q_W.to(torch::kFloat32), hca_k_W.to(torch::kFloat32),
            hca_v_W.to(torch::kFloat32), num_heads, d_hca_pre,
            d_hca_q_W, d_hca_k_W, d_hca_v_W, d_x);

        if (d_x_sorted_packed.defined() && d_x_sorted_packed.numel() > 0)
            d_x_sorted_packed.narrow(0, start, n).add_(d_x);
    }
    (void)d_model; (void)indexer_rank;
}


// ═══════════════════════════════════════════════════════════════════════
//  MoE/Adam multi-tensor — folded in from former launch_moe_adam.hip.cpp.
//
//  Caller passes pre-gathered active expert parameters. Math is identical
//  to AdamW (no SG2 metanet involvement on the multi-tensor path).
// ═══════════════════════════════════════════════════════════════════════

void launch_moe_adam_step(
    std::vector<torch::Tensor>& params,
    std::vector<torch::Tensor>& exp_avgs,
    std::vector<torch::Tensor>& exp_avg_sqs,
    std::vector<torch::Tensor>& grads,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        auto& p = params[i];
        auto& g = grads[i];
        auto& m = exp_avgs[i];
        auto& v = exp_avg_sqs[i];

        prim::ema_update_inplace(m, g, beta1);
        prim::ema_sq_update_inplace(v, g, beta2);
        prim::adam_apply_inplace(p, m, v, lr, bc1, bc2, eps, wd);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  MoE (Mixture of Experts) — folded in from former launch_moe.hip.cpp.
//
//  REAL gfx942 implementations of the MoE-compaction tail of
//  MoEAwareSuperGrok2 (Stage 1B). Because `.hip.cpp` TUs route through the
//  HOST compiler we cannot author __global__ kernels here; all work is ATen
//  tensor ops (which reach rocBLAS / rocPRIM internally), semantically mirroring
//  the sm_90 CUDA path in supergrok2_sm90.cuh.
//
//  Reachability (verified): Python's _moe_step calls count_expert_activations,
//  compute_load_balance_loss, apply_frequency_scaling, filter_active_params,
//  scatter_results. dynamic_expert_{load,fwd,bwd} and scan_compacted are
//  exported for ABI but not currently called; still implemented correctly.
//  moe_scan_compacted is VESTIGIAL (Mamba-era; SG2's mixer is now CSA/HCA).
// ═══════════════════════════════════════════════════════════════════════

// ── (1) Expert-activation histogram ──
//  expert_counts[e] += #rows with gate_logits[:,e] > threshold (int32).
void moe_count_expert_activations(
    torch::Tensor gate_logits, torch::Tensor expert_counts,
    float threshold, int N, int num_experts) {
    if (N == 0 || num_experts == 0) return;
    auto gl = gate_logits.to(torch::kFloat32);
    auto counts = (gl > threshold).sum(/*dim=*/0).to(torch::kInt32);  // [E]
    expert_counts.copy_(counts);
}

// ── (2) Switch-Transformer load-balance auxiliary loss ──
//  loss = E * Σ_e f_e * P_e ; f_e = counts[e]/N ; P_e = mean_t softmax(gl)[t,e].
torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor expert_counts, torch::Tensor gate_logits,
    int N, int num_experts) {
    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32).device(gate_logits.device());
    if (N == 0 || num_experts == 0) return torch::zeros({}, opts);
    auto gl = gate_logits.to(torch::kFloat32);
    auto P = torch::softmax(gl, /*dim=*/1).mean(/*dim=*/0);              // [E]
    auto f = expert_counts.to(torch::kFloat32) / static_cast<double>(N); // [E]
    return static_cast<double>(num_experts) * (f * P).sum();
}

// ── (3) Frequency-inverse per-expert LR scaling ──
//  freq_e = (counts[e]+s)/(total+s*E) ; scale_e = clamp((1/E)/freq_e, lo, hi).
void moe_apply_frequency_scaling(
    torch::Tensor expert_counts, torch::Tensor lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing) {
    if (num_experts == 0) return;
    const double denom = static_cast<double>(total_activations)
                       + static_cast<double>(smoothing) * num_experts;
    const double uniform = 1.0 / static_cast<double>(num_experts);
    auto freq = (expert_counts.to(torch::kFloat32) + smoothing) / denom;  // [E]
    auto scale = torch::clamp(uniform / freq, min_scale, max_scale);      // [E]
    lr_scale.copy_(scale.to(torch::kFloat32));
}

// ── (4) Stream-compaction of active-expert parameters ──
//  Keep index i iff expert_active[param_to_expert[i]] != 0. We build a boolean
//  mask and use it to gather the kept elements (deterministic ascending order,
//  unlike the sm_90 atomic compaction — both satisfy the (out->i) scatter
//  contract). compact_count[0] = #kept; outputs filled in [0, kept).
void moe_filter_active_params(
    torch::Tensor params, torch::Tensor grads,
    torch::Tensor state_m, torch::Tensor state_v,
    torch::Tensor param_to_expert, torch::Tensor expert_active,
    torch::Tensor compact_params, torch::Tensor compact_grads,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices, torch::Tensor compact_count,
    int total_params) {
    if (total_params == 0) {
        compact_count.zero_();
        return;
    }
    auto p2e = param_to_expert.to(torch::kLong);                  // [P]
    auto active = expert_active.to(torch::kBool);                 // [E]
    auto keep = active.index_select(0, p2e);                      // [P] bool
    auto idx = torch::nonzero(keep).reshape(-1);                  // [K] long
    const int64_t K = idx.numel();
    compact_count.fill_(static_cast<int>(K));
    if (K == 0) return;
    compact_params.narrow(0, 0, K).copy_(params.index_select(0, idx));
    compact_grads.narrow(0, 0, K).copy_(grads.index_select(0, idx));
    compact_state_m.narrow(0, 0, K).copy_(state_m.index_select(0, idx));
    compact_state_v.narrow(0, 0, K).copy_(state_v.index_select(0, idx));
    scatter_indices.narrow(0, 0, K).copy_(idx.to(torch::kInt32));
}

// ── (5) Scatter compacted results back to dense storage ──
//  params[scatter_indices[j]] = compact_params[j], etc. (inverse of (4)).
void moe_scatter_results(
    torch::Tensor compact_params,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices,
    torch::Tensor params,
    torch::Tensor state_m, torch::Tensor state_v,
    int compact_N) {
    if (compact_N == 0) return;
    auto idx = scatter_indices.narrow(0, 0, compact_N).to(torch::kLong);
    params.index_copy_(0, idx, compact_params.narrow(0, 0, compact_N));
    state_m.index_copy_(0, idx, compact_state_m.narrow(0, 0, compact_N));
    state_v.index_copy_(0, idx, compact_state_v.narrow(0, 0, compact_N));
}

// ── (6) Masked gather of active expert weights ──
//  Pack the slices of expert_{w1,b1,w2,b2} where active_mask[e]!=0, in
//  ascending expert order, into the smem_* buffers.
void moe_dynamic_expert_load(
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor active_mask,
    torch::Tensor smem_w1, torch::Tensor smem_b1,
    torch::Tensor smem_w2, torch::Tensor smem_b2) {
    auto idx = torch::nonzero(active_mask.reshape(-1) != 0).reshape(-1);  // [A]
    const int64_t A = idx.numel();
    if (A == 0) return;
    smem_w1.narrow(0, 0, A).copy_(expert_w1.index_select(0, idx));
    smem_w2.narrow(0, 0, A).copy_(expert_w2.index_select(0, idx));
    smem_b1.narrow(0, 0, A).copy_(expert_b1.index_select(0, idx));
    smem_b2.narrow(0, 0, A).copy_(expert_b2.index_select(0, idx));
}

// ── (7) Per-token expert MLP forward ──
//  output[t] = rw[t] * (W2_e @ relu(W1_e @ input[t] + b1_e) + b2_e).
//  Shapes: input [N,d_in]; expert_w1 [E,hidden,d_in]; expert_w2 [E,d_out,hidden].
//  Vectorized with batched bmm over per-token gathered expert weights.
torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor input, torch::Tensor expert_indices,
    torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor output) {
    const int64_t N = input.size(0);
    if (N == 0) return output;
    auto x   = input.to(torch::kFloat32);                       // [N,d_in]
    auto idx = expert_indices.to(torch::kLong);                 // [N]
    auto rw  = routing_weights.to(torch::kFloat32).reshape({N, 1});
    auto W1  = expert_w1.to(torch::kFloat32).index_select(0, idx);  // [N,H,d_in]
    auto b1  = expert_b1.to(torch::kFloat32).index_select(0, idx);  // [N,H]
    auto W2  = expert_w2.to(torch::kFloat32).index_select(0, idx);  // [N,d_out,H]
    auto b2  = expert_b2.to(torch::kFloat32).index_select(0, idx);  // [N,d_out]
    // h = relu(W1 @ x + b1)  -> [N,H]
    auto h = torch::relu(
        torch::bmm(W1, x.unsqueeze(2)).squeeze(2) + b1);           // [N,H]
    auto y = torch::bmm(W2, h.unsqueeze(2)).squeeze(2) + b2;       // [N,d_out]
    output.copy_(rw * y);
    return output;
}

// ── (8) Per-token expert MLP backward (full VJP) ──
//  dy = rw * d_output ; db2_e += dy ; dW2_e += dy⊗h ; dh = W2ᵀdy ;
//  dz1 = dh ⊙ [z1>0] ; dW1_e += dz1⊗x ; db1_e += dz1 ; d_input = W1ᵀ dz1.
//  Expert-weight grads accumulated via index_add_ (multiple tokens per expert).
void moe_dynamic_expert_bwd(
    torch::Tensor d_output, torch::Tensor input,
    torch::Tensor expert_indices, torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor d_input, torch::Tensor d_expert_w1,
    torch::Tensor d_expert_b1, torch::Tensor d_expert_w2,
    torch::Tensor d_expert_b2) {
    const int64_t N = input.size(0);
    if (N == 0) return;
    auto x   = input.to(torch::kFloat32);                          // [N,d_in]
    auto dout = d_output.to(torch::kFloat32);                      // [N,d_out]
    auto idx = expert_indices.to(torch::kLong);                    // [N]
    auto rw  = routing_weights.to(torch::kFloat32).reshape({N, 1});
    auto W1g = expert_w1.to(torch::kFloat32).index_select(0, idx); // [N,H,d_in]
    auto b1g = expert_b1.to(torch::kFloat32).index_select(0, idx); // [N,H]
    auto W2g = expert_w2.to(torch::kFloat32).index_select(0, idx); // [N,d_out,H]

    // Recompute forward activations.
    auto z1 = torch::bmm(W1g, x.unsqueeze(2)).squeeze(2) + b1g;    // [N,H]
    auto relu_mask = (z1 > 0).to(torch::kFloat32);                 // [N,H]
    auto h = z1 * relu_mask;                                       // [N,H]

    auto dy = rw * dout;                                           // [N,d_out]
    // dW2 = dy ⊗ h : [N,d_out,H] ; db2 = dy ; dh = W2ᵀ dy.
    auto dW2 = torch::bmm(dy.unsqueeze(2), h.unsqueeze(1));        // [N,d_out,H]
    auto db2 = dy;                                                 // [N,d_out]
    auto dh = torch::bmm(W2g.transpose(1, 2), dy.unsqueeze(2)).squeeze(2); // [N,H]
    auto dz1 = dh * relu_mask;                                     // [N,H]
    auto dW1 = torch::bmm(dz1.unsqueeze(2), x.unsqueeze(1));       // [N,H,d_in]
    auto db1 = dz1;                                                // [N,H]
    auto dx = torch::bmm(W1g.transpose(1, 2), dz1.unsqueeze(2)).squeeze(2); // [N,d_in]
    d_input.copy_(dx);

    // Scatter-accumulate per-expert grads.
    d_expert_w1.index_add_(0, idx, dW1);
    d_expert_b1.index_add_(0, idx, db1);
    d_expert_w2.index_add_(0, idx, dW2);
    d_expert_b2.index_add_(0, idx, db2);
}

// ── (9) Compacted selective scan — VESTIGIAL ──
//  Mamba-era discretized SSM recurrence; SG2's mixer is now CSA/HCA and Python
//  NEVER calls this. Kept linkable + numerically sound for ABI stability.
//    A = -exp(A_log) ; A_bar = exp(dt*A) ; h_t = A_bar*h_{t-1} + (dt*B_t)*x_t ;
//    y_t = Σ_s C_t[s]*h_t[d,s] + D[d]*x_t[d].
//  Layout: compact_x/dt [Nc,d_inner]; compact_B/C [Nc,d_state];
//          A_log [d_inner,d_state]; D_param [d_inner];
//          initial/final_state [d_inner,d_state]; scan_output [Nc,d_inner].
void moe_scan_compacted(
    torch::Tensor compact_x, torch::Tensor compact_dt,
    torch::Tensor compact_B, torch::Tensor compact_C,
    torch::Tensor A_log, torch::Tensor D_param,
    torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int compact_N, int d_inner, int d_state) {
    (void)rope_freq;  // vestigial positional arg (Mamba-era), intentionally unused.
    if (compact_N == 0 || d_inner == 0) return;
    auto x  = compact_x.to(torch::kFloat32);     // [Nc,d_inner]
    auto dt = compact_dt.to(torch::kFloat32);    // [Nc,d_inner]
    auto B  = compact_B.to(torch::kFloat32);     // [Nc,d_state]
    auto C  = compact_C.to(torch::kFloat32);     // [Nc,d_state]
    auto A  = -torch::exp(A_log.to(torch::kFloat32));  // [d_inner,d_state]
    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32).device(x.device());
    auto h = (initial_state.defined() && initial_state.numel() > 0)
           ? initial_state.to(torch::kFloat32).clone()
           : torch::zeros({d_inner, d_state}, opts);     // [d_inner,d_state]
    auto Dvec = (D_param.defined() && D_param.numel() > 0)
              ? D_param.to(torch::kFloat32)
              : torch::zeros({d_inner}, opts);            // [d_inner]
    for (int t = 0; t < compact_N; ++t) {
        auto xt = x[t].reshape({d_inner, 1});             // [d_inner,1]
        auto dtt = dt[t].reshape({d_inner, 1});           // [d_inner,1]
        auto Bt = B[t].reshape({1, d_state});             // [1,d_state]
        auto Ct = C[t].reshape({1, d_state});             // [1,d_state]
        auto A_bar = torch::exp(dtt * A);                 // [d_inner,d_state]
        h = A_bar * h + (dtt * Bt) * xt;                  // [d_inner,d_state]
        auto y = (h * Ct).sum(/*dim=*/1) + Dvec * x[t];   // [d_inner]
        scan_output[t].copy_(y);
    }
    if (final_state.defined() && final_state.numel() > 0)
        final_state.copy_(h);
}

}} // namespace sg::gfx942

#endif  // GROKKING_KERNELS_GFX942_SUPERGROK2_GFX942_HIP_HPP_
