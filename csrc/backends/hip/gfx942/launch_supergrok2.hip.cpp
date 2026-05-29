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

// ── inlined from former csrc/backends/hip/gfx942/primitives.hpp ──
// HIP gfx942 (CDNA3 / MI300X) primitives — host-side ATen helpers shared
// across every HIP launch file.

namespace sg { namespace gfx942 { namespace primitives {

inline void check_device(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be on a HIP/CUDA device");
}

template <typename... Tensors>
inline bool keep_tensor(const torch::Tensor& grad) {
    return grad.defined() && grad.numel() > 0;
}

inline void ema_update_inplace(
    torch::Tensor& m, const torch::Tensor& g, float beta1
) {
    m.mul_(beta1).add_(g, 1.0f - beta1);
}

inline void ema_sq_update_inplace(
    torch::Tensor& v, const torch::Tensor& g, float beta2
) {
    v.mul_(beta2).addcmul_(g, g, 1.0f - beta2);
}

inline void adam_apply_inplace(
    torch::Tensor& p, const torch::Tensor& m, const torch::Tensor& v,
    float lr, float bc1, float bc2, float eps, float wd
) {
    auto m_hat = m / bc1;  // bc1 = 1 - beta1^t (un-inverted)
    auto v_hat = v / bc2;  // bc2 = 1 - beta2^t (un-inverted)
    auto denom = v_hat.sqrt().add_(eps);
    auto update = m_hat.div_(denom).add_(p, wd);
    p.add_(update, -lr);
}

struct TensorPack {
    std::vector<torch::Tensor> params;
    std::vector<torch::Tensor> grads;
    std::vector<torch::Tensor> state_a;
    std::vector<torch::Tensor> state_b;
};

inline TensorPack pack_valid(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const std::vector<torch::Tensor>& state_a = {},
    const std::vector<torch::Tensor>& state_b = {}
) {
    TensorPack out;
    for (size_t i = 0; i < params.size(); i++) {
        if (!grads[i].defined() || grads[i].numel() == 0) continue;
        out.params.push_back(params[i]);
        out.grads.push_back(grads[i]);
        if (!state_a.empty()) out.state_a.push_back(state_a[i]);
        if (!state_b.empty()) out.state_b.push_back(state_b[i]);
    }
    return out;
}

}}} // namespace sg::gfx942::primitives
// ── end inlined csrc/backends/hip/gfx942/primitives.hpp ──

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
//  Bilevel forward-save + backward.
//
//  HONEST STATUS: NOT implemented on gfx942 (spec §7 permits this as a throw).
//  The forward step / batched step is functional. The bilevel saved-activations
//  adjoint for CSA/HCA (softmax denominators + selected-index sets) is not
//  written here; callers must run bilevel on sm_90 or use the forward-only
//  `supergrok2_prepare_and_batched_step` entry point.
// ═══════════════════════════════════════════════════════════════════════

[[noreturn]] static void csa_hca_bilevel_not_implemented(const char* op) {
    throw std::runtime_error(
        std::string("csa_hca bilevel ") + op + " not implemented on gfx942. "
        "The forward path (launch_csa_hca_step / launch_csa_hca_batched_step) is "
        "functional; only the saved-activations bilevel backward path is missing. "
        "Use the forward-only `supergrok2_prepare_and_batched_step` entry point, "
        "or run bilevel on sm_90.");
}

void launch_csa_hca_bilevel_fwd_save(...) {
    csa_hca_bilevel_not_implemented("fwd_save");
}

void launch_csa_hca_bilevel_fwd_save_batched(...) {
    csa_hca_bilevel_not_implemented("fwd_save_batched");
}

void launch_csa_hca_backward(...) {
    csa_hca_bilevel_not_implemented("backward");
}

void launch_csa_hca_backward_batched(...) {
    csa_hca_bilevel_not_implemented("backward_batched");
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
//  Placeholder kernels that throw at runtime. Real gfx942 MFMA-fused expert
//  MLP kernels are the obvious follow-up (spec §9: "MoE for SG2 should be REAL
//  where feasible"); the host-compiled ATen TU keeps them as stubs for now so
//  the binding symbols resolve.
// ═══════════════════════════════════════════════════════════════════════

void moe_dynamic_expert_load(
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor active_mask,
    torch::Tensor smem_w1, torch::Tensor smem_b1,
    torch::Tensor smem_w2, torch::Tensor smem_b2) {
    throw std::runtime_error(
        "moe_dynamic_expert_load: gfx942 kernel not yet implemented.");
}

torch::Tensor moe_dynamic_expert_fwd(
    torch::Tensor input, torch::Tensor expert_indices,
    torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor output) {
    throw std::runtime_error(
        "moe_dynamic_expert_fwd: gfx942 kernel not yet implemented.");
    return torch::Tensor{};
}

void moe_dynamic_expert_bwd(
    torch::Tensor d_output, torch::Tensor input,
    torch::Tensor expert_indices, torch::Tensor routing_weights,
    torch::Tensor expert_w1, torch::Tensor expert_b1,
    torch::Tensor expert_w2, torch::Tensor expert_b2,
    torch::Tensor d_input, torch::Tensor d_expert_w1,
    torch::Tensor d_expert_b1, torch::Tensor d_expert_w2,
    torch::Tensor d_expert_b2) {
    throw std::runtime_error(
        "moe_dynamic_expert_bwd: gfx942 kernel not yet implemented.");
}

void moe_filter_active_params(
    torch::Tensor params, torch::Tensor grads,
    torch::Tensor state_m, torch::Tensor state_v,
    torch::Tensor param_to_expert, torch::Tensor expert_active,
    torch::Tensor compact_params, torch::Tensor compact_grads,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices, torch::Tensor compact_count,
    int total_params) {
    throw std::runtime_error(
        "moe_filter_active_params: gfx942 kernel not yet implemented.");
}

void moe_scan_compacted(
    torch::Tensor compact_x, torch::Tensor compact_dt,
    torch::Tensor compact_B, torch::Tensor compact_C,
    torch::Tensor A_log, torch::Tensor D_param,
    torch::Tensor rope_freq,
    torch::Tensor scan_output, torch::Tensor final_state,
    torch::Tensor initial_state,
    int compact_N, int d_inner, int d_state) {
    throw std::runtime_error(
        "moe_scan_compacted: gfx942 kernel not yet implemented.");
}

void moe_scatter_results(
    torch::Tensor compact_params,
    torch::Tensor compact_state_m, torch::Tensor compact_state_v,
    torch::Tensor scatter_indices,
    torch::Tensor params,
    torch::Tensor state_m, torch::Tensor state_v,
    int compact_N) {
    throw std::runtime_error(
        "moe_scatter_results: gfx942 kernel not yet implemented.");
}

void moe_count_expert_activations(
    torch::Tensor gate_logits, torch::Tensor expert_counts,
    float threshold, int N, int num_experts) {
    throw std::runtime_error(
        "moe_count_expert_activations: gfx942 kernel not yet implemented.");
}

torch::Tensor moe_compute_load_balance_loss(
    torch::Tensor expert_counts, torch::Tensor gate_logits,
    int N, int num_experts) {
    throw std::runtime_error(
        "moe_compute_load_balance_loss: gfx942 kernel not yet implemented.");
    return torch::Tensor{};
}

void moe_apply_frequency_scaling(
    torch::Tensor expert_counts, torch::Tensor lr_scale,
    int num_experts, int total_activations,
    float min_scale, float max_scale, float smoothing) {
    throw std::runtime_error(
        "moe_apply_frequency_scaling: gfx942 kernel not yet implemented.");
}

}} // namespace sg::gfx942
