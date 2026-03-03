/*
 * LibTorch Model Definitions
 *
 * Layer 3: C++ equivalents of the Python model architectures.
 * These produce identical outputs given identical weights.
 *
 * Models:
 *   1. Transformer (Decoder) — causal attention, (a ÷ b) mod p
 *   2. ViT (Encoder)         — full attention + CLS token, MNIST addition mod p
 *   3. MambaModel (SSM)      — selective state space, sequential division mod p
 *
 * All models use LibTorch's autograd for backward passes.
 * Forward passes use standard ATen operations (which dispatch to cuBLAS/cuDNN).
 * Future: replace with custom fused kernels for attention+MLP.
 */
#pragma once

#include <torch/torch.h>
#include <cmath>


// ═══════════════════════════════════════════════════════════════════════
//  Decoder Block (causal self-attention + FFN)
// ═══════════════════════════════════════════════════════════════════════

struct DecoderBlockImpl : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm n1{nullptr}, n2{nullptr};
    torch::nn::Sequential ff{nullptr};

    DecoderBlockImpl(int d, int h)
        : attn(register_module("attn",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(d, h)
                    .dropout(0.0).batch_first(true)))),
          n1(register_module("n1", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({d})))),
          n2(register_module("n2", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({d})))),
          ff(register_module("ff", torch::nn::Sequential(
              torch::nn::Linear(d, 4 * d),
              torch::nn::GELU(),
              torch::nn::Linear(4 * d, d))))
    {}

    torch::Tensor forward(torch::Tensor x) {
        int seq_len = x.size(1);
        auto mask = torch::triu(
            torch::ones({seq_len, seq_len},
                torch::TensorOptions().dtype(torch::kBool).device(x.device())),
            1);
        auto [a, _] = attn(x, x, x, /*key_padding_mask=*/{}, /*need_weights=*/false, mask);
        x = n1(x + a);
        return n2(x + ff->forward(x));
    }
};
TORCH_MODULE(DecoderBlock);


// ═══════════════════════════════════════════════════════════════════════
//  Decoder Transformer — (a ÷ b) mod p
// ═══════════════════════════════════════════════════════════════════════

struct TransformerImpl : torch::nn::Module {
    torch::nn::Embedding tok{nullptr}, pos{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear out{nullptr};

    TransformerImpl(int nl = 2, int d = 128, int h = 4, int ntok = 99, int seq = 4)
        : tok(register_module("tok", torch::nn::Embedding(ntok, d))),
          pos(register_module("pos", torch::nn::Embedding(seq, d))),
          layers(register_module("layers", torch::nn::ModuleList())),
          norm(register_module("norm", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({d})))),
          out(register_module("out", torch::nn::Linear(d, ntok)))
    {
        for (int i = 0; i < nl; i++) {
            layers->push_back(DecoderBlock(d, h));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto positions = torch::arange(x.size(1),
            torch::TensorOptions().device(x.device())).unsqueeze(0);
        auto h = tok(x) + pos(positions);
        for (size_t i = 0; i < layers->size(); i++) {
            h = layers[i]->as<DecoderBlock>()->forward(h);
        }
        // Take last token's output
        return out(norm(h).select(1, -1));
    }
};
TORCH_MODULE(Transformer);


// ═══════════════════════════════════════════════════════════════════════
//  Encoder Block (full self-attention + FFN, no causal mask)
// ═══════════════════════════════════════════════════════════════════════

struct EncoderBlockImpl : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm n1{nullptr}, n2{nullptr};
    torch::nn::Sequential ff{nullptr};

    EncoderBlockImpl(int d, int h)
        : attn(register_module("attn",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(d, h)
                    .dropout(0.0).batch_first(true)))),
          n1(register_module("n1", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({d})))),
          n2(register_module("n2", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({d})))),
          ff(register_module("ff", torch::nn::Sequential(
              torch::nn::Linear(d, 4 * d),
              torch::nn::GELU(),
              torch::nn::Linear(4 * d, d))))
    {}

    torch::Tensor forward(torch::Tensor x) {
        auto [a, _] = attn(x, x, x);
        x = n1(x + a);
        return n2(x + ff->forward(x));
    }
};
TORCH_MODULE(EncoderBlock);


// ═══════════════════════════════════════════════════════════════════════
//  Vision Transformer — MNIST (a + b) mod p
// ═══════════════════════════════════════════════════════════════════════

struct ViTImpl : torch::nn::Module {
    torch::nn::Linear patch_proj{nullptr};
    torch::Tensor cls_token;
    torch::nn::Embedding pos_embed{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear out{nullptr};

    ViTImpl(int p = 97, int patch_dim = 49, int num_patches = 16,
            int d = 128, int h = 4, int nl = 2)
        : patch_proj(register_module("patch_proj", torch::nn::Linear(patch_dim, d))),
          cls_token(register_parameter("cls_token",
              torch::randn({1, 1, d}) * 0.02)),
          pos_embed(register_module("pos", torch::nn::Embedding(num_patches + 1, d))),
          layers(register_module("layers", torch::nn::ModuleList())),
          norm(register_module("norm", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({d})))),
          out(register_module("out", torch::nn::Linear(d, p)))
    {
        for (int i = 0; i < nl; i++) {
            layers->push_back(EncoderBlock(d, h));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        int B = x.size(0);
        auto h = patch_proj(x);
        h = torch::cat({cls_token.expand({B, -1, -1}), h}, 1);
        auto positions = torch::arange(h.size(1),
            torch::TensorOptions().device(x.device())).unsqueeze(0);
        h = h + pos_embed(positions);
        for (size_t i = 0; i < layers->size(); i++) {
            h = layers[i]->as<EncoderBlock>()->forward(h);
        }
        // CLS token output
        return out(norm(h.select(1, 0)));
    }
};
TORCH_MODULE(ViT);


// ═══════════════════════════════════════════════════════════════════════
//  Selective SSM Layer (Mamba-style)
// ═══════════════════════════════════════════════════════════════════════

struct SelectiveSSMLayerImpl : torch::nn::Module {
    int state_dim, d_inner, dt_rank;
    torch::nn::Linear in_proj{nullptr}, x_proj{nullptr}, dt_proj{nullptr}, out_proj{nullptr};
    torch::nn::Conv1d conv1d{nullptr};
    torch::Tensor A_log, D_param;
    torch::nn::LayerNorm norm{nullptr};

    SelectiveSSMLayerImpl(int d, int state_dim_ = 16, int dt_rank_ = -1, int expand_factor = 2)
        : state_dim(state_dim_),
          d_inner(d * expand_factor),
          dt_rank(dt_rank_ > 0 ? dt_rank_ : std::max(d / 16, 1))
    {
        in_proj = register_module("in_proj",
            torch::nn::Linear(torch::nn::LinearOptions(d, d_inner * 2).bias(false)));
        conv1d = register_module("conv1d",
            torch::nn::Conv1d(torch::nn::Conv1dOptions(d_inner, d_inner, 3)
                .padding(1).groups(d_inner).bias(true)));
        x_proj = register_module("x_proj",
            torch::nn::Linear(torch::nn::LinearOptions(d_inner, dt_rank + state_dim * 2).bias(false)));
        dt_proj = register_module("dt_proj",
            torch::nn::Linear(torch::nn::LinearOptions(dt_rank, d_inner).bias(true)));

        auto A = torch::arange(1, state_dim + 1, torch::kFloat32);
        A_log = register_parameter("A_log",
            torch::log(A.unsqueeze(0).expand({d_inner, -1}).clone()));
        D_param = register_parameter("D", torch::ones(d_inner));

        out_proj = register_module("out_proj",
            torch::nn::Linear(torch::nn::LinearOptions(d_inner, d).bias(false)));
        norm = register_module("norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d})));
    }

    torch::Tensor selective_scan(torch::Tensor x, torch::Tensor dt,
                                  torch::Tensor B, torch::Tensor C) {
        int batch = x.size(0), L = x.size(1);
        auto A = -torch::exp(A_log);
        dt = torch::softplus(dt);

        auto h = torch::zeros({batch, d_inner, state_dim},
            torch::TensorOptions().dtype(x.dtype()).device(x.device()));

        std::vector<torch::Tensor> ys;
        ys.reserve(L);
        for (int t = 0; t < L; t++) {
            auto dt_t = dt.select(1, t).unsqueeze(-1);
            h = torch::exp(dt_t * A.unsqueeze(0)) * h
                + (dt_t * B.select(1, t).unsqueeze(1)) * x.select(1, t).unsqueeze(-1);
            ys.push_back((h * C.select(1, t).unsqueeze(1)).sum(-1));
        }
        return torch::stack(ys, 1);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        auto xz = in_proj(x);
        auto chunks = xz.chunk(2, -1);
        auto x_main = chunks[0], z = chunks[1];
        x_main = torch::silu(
            conv1d(x_main.transpose(1, 2)).transpose(1, 2));
        auto x_dbc = x_proj(x_main);
        auto splits = x_dbc.split({dt_rank, state_dim, state_dim}, -1);
        auto y = selective_scan(x_main, dt_proj(splits[0]), splits[1], splits[2]);
        y = out_proj(
            (y + x_main * D_param.unsqueeze(0).unsqueeze(0)) * torch::silu(z));
        return norm(y + residual);
    }
};
TORCH_MODULE(SelectiveSSMLayer);


// ═══════════════════════════════════════════════════════════════════════
//  Mamba Model — (a÷b₁÷b₂÷b₃) mod p
// ═══════════════════════════════════════════════════════════════════════

struct MambaModelImpl : torch::nn::Module {
    torch::nn::Embedding tok{nullptr}, pos_embed{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear out{nullptr};

    MambaModelImpl(int p = 97, int ntok = 99, int seq_len = 8,
                   int d = 128, int nl = 2)
        : tok(register_module("tok", torch::nn::Embedding(ntok, d))),
          pos_embed(register_module("pos", torch::nn::Embedding(seq_len, d))),
          layers(register_module("layers", torch::nn::ModuleList())),
          norm(register_module("norm", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({d})))),
          out(register_module("out", torch::nn::Linear(d, p)))
    {
        for (int i = 0; i < nl; i++) {
            layers->push_back(SelectiveSSMLayer(d));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto positions = torch::arange(x.size(1),
            torch::TensorOptions().device(x.device())).unsqueeze(0);
        auto h = tok(x) + pos_embed(positions);
        for (size_t i = 0; i < layers->size(); i++) {
            h = layers[i]->as<SelectiveSSMLayer>()->forward(h);
        }
        // Last token output
        return out(norm(h).select(1, -1));
    }
};
TORCH_MODULE(MambaModel);


// ═══════════════════════════════════════════════════════════════════════
//  Model Factory
// ═══════════════════════════════════════════════════════════════════════

inline torch::nn::AnyModule build_model(
    const std::string& model_type,
    int num_layers, int d_model, int num_heads,
    int num_tokens, int p, int seq_len,
    int patch_dim = 49, int num_patches = 16
) {
    if (model_type == "decoder") {
        return torch::nn::AnyModule(
            Transformer(num_layers, d_model, num_heads, num_tokens, seq_len));
    } else if (model_type == "vit") {
        return torch::nn::AnyModule(
            ViT(p, patch_dim, num_patches, d_model, num_heads, num_layers));
    } else if (model_type == "mamba") {
        return torch::nn::AnyModule(
            MambaModel(p, num_tokens, seq_len, d_model, num_layers));
    }
    throw std::runtime_error("Unknown model type: " + model_type);
}
