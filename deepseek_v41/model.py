"""DeepSeek-V4.1-Flash text backbone, rewritten for training in plain PyTorch.

This is a port of the official inference reference (``inference/model.py`` of
deepseek-ai/DeepSeek-V4.1-Flash). The maths of every module is kept; what
changes is the execution model:

* Full-sequence (prefill) path only. The decode path, KV caches and ring
  buffers are gone; causality is enforced with explicit masks.
* The tilelang kernels are replaced by differentiable PyTorch:
  ``sparse_attn`` becomes masked dense attention with the same per-head
  "attention sink" logit, and ``hc_split_sinkhorn`` is written out in torch.
  FP8/FP4 weight and activation quantization is not simulated (weights are
  ordinary fp32 parameters; run under bf16 autocast for mixed precision).
* The process-global ``SharedAttentionRuntime`` becomes a per-forward context,
  so repeated forwards (evaluation, sharpness-aware and meta steps) cannot leak
  state into each other.
* In-place RoPE is replaced by an out-of-place version so autograd works.
* The vision tower and the DSpark draft layers are not included.
* Training-only pieces the inference code does not need are added: parameter
  initialization, MoE load tracking and the auxiliary-loss-free router-bias
  update (:meth:`Transformer.update_router_bias`), parameter-role metadata for
  optimizers (:meth:`Transformer.param_roles`) and FLOP accounting.

Dense masked attention costs O(S x (S + S/ratio)) memory per head. That is fine
for the short grokking sequences; long-context training needs the sparse
kernels, which belong to the CUDA phase.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import Config

# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def fp32_region(x: torch.Tensor):
    """Disable autocast for parts the reference computes in fp32 (router, compressor, mHC, head)."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.autocast(x.device.type, enabled=False)
    return nullcontext()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Rotary frequencies as unit complex numbers, one row per position (YaRN when original_seq_len > 0)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent element pairs of ``x`` ([b, s, d] or [b, s, h, d]); out of place."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs_cis.conj() if inverse else freqs_cis
    f = f.view(1, x.size(1), xc.size(-1)) if x.ndim == 3 else f.view(1, x.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * f).flatten(-2).to(x.dtype)


def rope_tail(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """RoPE on the last ``rd`` channels only (the reference's ``x[..., -rd:]`` convention)."""
    return torch.cat([x[..., :-rd], apply_rotary_emb(x[..., -rd:], freqs_cis, inverse)], dim=-1)


def sink_attention(q, kv, mask, sink, scale):
    """Dense equivalent of the reference ``sparse_attn`` kernel.

    q [b, s, h, d]; kv [b, n, d] is both key and value (single latent KV head);
    mask [b, s, n] marks the positions each query may use; sink [h] is a
    per-head logit that joins the softmax normalizer but carries no value. A
    query with no visible position therefore outputs zeros, as in the kernel.
    """
    with fp32_region(q):  # einsum -> bmm, which autocast would otherwise run in bf16
        scores = torch.einsum("bshd,bnd->bshn", q.float(), kv.float()) * scale
        scores = scores.masked_fill(~mask[:, :, None, :], float("-inf"))
        b, s, h, _ = scores.shape
        logits = torch.cat([scores, sink.float().view(1, 1, h, 1).expand(b, s, h, 1)], dim=-1)
        probs = logits.softmax(dim=-1)[..., :-1]
        out = torch.einsum("bshn,bnd->bshd", probs, kv.float())
    return out.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc: int, iters: int, eps: float):
    """Torch version of the reference ``hc_split_sinkhorn`` kernel.

    mixes [..., (2 + hc) * hc] -> pre [..., hc], post [..., hc], comb [..., hc, hc]
    with comb pushed toward a doubly-stochastic matrix by Sinkhorn iterations.
    """
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


@dataclass
class AttnContext:
    """What attention layers hand down the stack in one forward (the reference's shared runtime)."""

    compress_kv: dict = field(default_factory=dict)  # kv source layer -> [b, n, head_dim] (post-RoPE)
    index_k: dict = field(default_factory=dict)  # kv source layer -> [b, n, index_head_dim]
    selected: dict = field(default_factory=dict)  # index source layer -> bool [b, s, n]
    candidates: torch.Tensor | None = None


# --------------------------------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------------------------------


class Compressor(nn.Module):
    """Pools ``ratio`` consecutive tokens into one KV latent with a learned per-channel softmax gate.

    Returns the pre-RoPE latents of the *complete* groups only; a trailing partial
    group is invisible to every query anyway (a group becomes visible once the
    query has passed its last token).
    """

    def __init__(self, cfg: Config, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False)
        self.wgate = nn.Linear(cfg.dim, cfg.head_dim, bias=False) if ratio > 1 else None

    def forward(self, x):
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        b, s, _ = x.shape
        n = s // self.ratio
        with fp32_region(x):
            xf = x[:, : n * self.ratio].float()
            kv = self.wkv(xf).unflatten(1, (n, self.ratio))
            score = self.wgate(xf).unflatten(1, (n, self.ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(kv.to(x.dtype))


def select_candidate_blocks(scores, compress_lens, topk_blocks: int, block_size: int):
    """Level one of the two-level top-k: keep the best ``topk_blocks`` blocks of compressed positions.

    scores [b, s, n] with unreachable positions at -inf; compress_lens [s, 1] (visible count per query).
    The block holding the query's newest position is always kept. Returns a bool mask like ``scores``.
    """
    width = scores.size(-1)
    padded = F.pad(scores, (0, -width % block_size), value=float("-inf"))
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    n_blocks = block_scores.size(-1)
    last = (compress_lens - 1) // block_size
    block_scores = block_scores.masked_fill(torch.arange(n_blocks, device=scores.device) == last, float("inf"))
    top = block_scores.topk(min(topk_blocks, n_blocks), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):
    """Scores compressed positions per query and keeps the ``index_topk`` best (CSA2 re-index step).

    Only layers that compress their own KV own the index keys (``wk``); later index
    sources reuse them with their own queries. The selection is a hard top-k, so
    the language-model loss sends no gradient here (see :meth:`index_scores`).
    """

    def __init__(self, cfg: Config, layer_id: int, owns_k: bool):
        super().__init__()
        self.ratio = cfg.compress_ratios[layer_id]
        self.owns_k = owns_k
        self.is_candidate_source = layer_id == cfg.candidate_source_layer
        self.uses_candidates = 0 <= cfg.candidate_source_layer < layer_id
        self.n_heads, self.head_dim, self.rd = cfg.index_n_heads, cfg.index_head_dim, cfg.rope_head_dim
        self.topk, self.cand_blocks, self.cand_block_size = (
            cfg.index_topk,
            cfg.candidate_topk_blocks,
            cfg.candidate_block_size,
        )
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.dim, self.n_heads, bias=False)
        if owns_k:
            self.wk = nn.Linear(cfg.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def make_keys(self, latent, freqs_c):
        k = self.k_norm(self.wk(latent))
        return rope_tail(k, freqs_c, self.rd)

    def index_scores(self, x, qr, index_k, freqs):
        """[b, s, n] relu-attention scores over compressed positions (before masking)."""
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rope_tail(q, freqs, self.rd)
        w = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)
        with fp32_region(q):
            s = torch.einsum("bshd,btd->bsht", q.float(), index_k.float()).relu()
            return (s * w.float().unsqueeze(-1)).sum(dim=2)

    @torch.no_grad()
    def select(self, scores, ctx: AttnContext):
        b, s, n = scores.shape
        compress_lens = (torch.arange(1, s + 1, device=scores.device) // self.ratio).unsqueeze(-1)
        visible = torch.arange(n, device=scores.device) < compress_lens  # [s, n]
        scores = scores.masked_fill(~visible, float("-inf"))
        if self.is_candidate_source:
            ctx.candidates = select_candidate_blocks(scores, compress_lens, self.cand_blocks, self.cand_block_size)
        elif self.uses_candidates and ctx.candidates is not None:
            scores = scores.masked_fill(~ctx.candidates, float("-inf"))
        k = min(self.topk, n)
        chosen = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, scores.topk(k, dim=-1).indices, True)
        return chosen & torch.isfinite(scores)  # never outside the visible range or the candidate pool


class Attention(nn.Module):
    """Latent MQA over a causal sliding window of raw KV plus (when the layer's ratio > 0) the
    index-selected compressed KV shared from the layer's kv source."""

    def __init__(self, cfg: Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads, self.head_dim, self.rd = cfg.n_heads, cfg.head_dim, cfg.rope_head_dim
        self.n_groups, self.o_lora_rank = cfg.o_groups, cfg.o_lora_rank
        self.window = cfg.window_size
        self.ratio = cfg.compress_ratios[layer_id]
        self.kv_source = cfg.kv_source(layer_id)
        self.index_source = cfg.index_source(layer_id)
        self.scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        # block-diagonal over groups: group g maps its own heads to o_lora_rank (applied with einsum)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, bias=False
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, cfg.dim, bias=False)

        self.is_kv_source = self.ratio > 0 and self.kv_source == layer_id
        self.is_index_source = self.ratio > 0 and self.index_source == layer_id
        self.compressor = Compressor(cfg, self.ratio) if self.is_kv_source else None
        self.indexer = Indexer(cfg, layer_id, owns_k=self.is_kv_source) if self.is_index_source else None
        if self.ratio:
            freqs = precompute_freqs_cis(
                self.rd,
                cfg.max_seq_len,
                cfg.original_seq_len,
                cfg.compress_rope_theta,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
            )
        else:  # pure sliding-window layers use the base theta and no YaRN
            freqs = precompute_freqs_cis(
                self.rd, cfg.max_seq_len, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow
            )
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, x, ctx: AttnContext):
        b, s, _ = x.shape
        if s > self.freqs_cis.size(0):
            raise ValueError(f"sequence length {s} exceeds max_seq_len {self.freqs_cis.size(0)}")
        freqs = self.freqs_cis[:s]
        qr = self.q_norm(self.wq_a(x))
        q = rope_tail(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), freqs, self.rd)

        kv = rope_tail(self.kv_norm(self.wkv(x)), freqs, self.rd)
        i = torch.arange(s, device=x.device)
        win_mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.window)
        mask = win_mask.expand(b, s, s)

        if self.ratio:
            if self.is_kv_source:
                latent = self.compressor(x)  # pre-RoPE, [b, n, head_dim]
                freqs_c = self.freqs_cis[: s - s % self.ratio : self.ratio]  # group j sits at token j * ratio
                ctx.index_k[self.layer_id] = self.indexer.make_keys(latent, freqs_c)
                ctx.compress_kv[self.layer_id] = rope_tail(latent, freqs_c, self.rd)
            compress_kv = ctx.compress_kv[self.kv_source]
            if self.is_index_source:
                scores = self.indexer.index_scores(x, qr, ctx.index_k[self.kv_source], freqs)
                ctx.selected[self.layer_id] = self.indexer.select(scores.detach(), ctx)
            selected = ctx.selected[self.index_source]
            kv = torch.cat([kv, compress_kv], dim=1)
            mask = torch.cat([mask, selected], dim=-1)

        o = sink_attention(q, kv, mask, self.attn_sink, self.scale)
        o = rope_tail(o, freqs, self.rd, inverse=True)
        o = o.reshape(b, s, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


# --------------------------------------------------------------------------------------------------
# mixture of experts
# --------------------------------------------------------------------------------------------------


class Gate(nn.Module):
    """Top-k router. The correction bias (a buffer, updated by a rule rather than by gradients) only
    decides which experts are chosen; the combine weights come from the unbiased scores."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.topk, self.score_func, self.temp = cfg.n_activated_experts, cfg.score_func, cfg.gate_temp
        self.norm_topk_prob, self.route_scale = cfg.norm_topk_prob, cfg.route_scale
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        self.register_buffer("bias", torch.zeros(cfg.n_routed_experts))
        self.register_buffer("load", torch.zeros(cfg.n_routed_experts), persistent=False)
        self.balance_loss: torch.Tensor | None = None
        self.track_load = True  # off inside optimizers' extra forwards (see Transformer.frozen_router_stats)

    def forward(self, x, seq_len: int | None = None):
        """x [tokens, dim] (tokens = batch * seq_len, sequence-major) -> (weights, indices)."""
        with fp32_region(x):
            scores = F.linear(x.float(), self.weight.float()) / self.temp
            if self.score_func == "softmax":
                scores = scores.softmax(dim=-1)
            elif self.score_func == "sigmoid":
                scores = scores.sigmoid()
            else:
                scores = F.softplus(scores).sqrt()
            indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
            weights = scores.gather(1, indices)
            if self.norm_topk_prob and self.topk > 1:
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            weights = weights * self.route_scale
        self.balance_loss = None
        if self.training and torch.is_grad_enabled():
            if self.track_load:
                self.load += torch.bincount(indices.flatten(), minlength=self.load.numel()).to(self.load.dtype)
            if seq_len:
                self.balance_loss = self.sequence_balance_loss(scores, indices, seq_len)
        return weights, indices

    def sequence_balance_loss(self, scores, indices, seq_len: int):
        """DeepSeek-V3 sequence-wise balance term sum_i f_i * P_i, averaged over sequences.

        P_i is the mean normalized affinity of expert i over the sequence (differentiable);
        f_i = E / (K * T) * (tokens routed to i), from the biased selection (constant).
        Equals 1 for perfectly uniform routing. Scaled by ``balance_loss_alpha`` in the model.
        """
        n_exp = scores.size(-1)
        probs = (scores / scores.sum(-1, keepdim=True)).view(-1, seq_len, n_exp)
        routed = torch.zeros_like(scores).scatter_(-1, indices, 1.0).view(-1, seq_len, n_exp)
        f = routed.sum(1) * (n_exp / (self.topk * seq_len))
        return (f.detach() * probs.mean(1)).sum(-1).mean()


class Expert(nn.Module):
    """SwiGLU FFN with the reference's activation clamps (up clamped both ways, gate from above)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x, weights=None):
        dtype = x.dtype
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.swiglu_limit > 0:
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


class MoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.dim = cfg.dim
        self.gate = Gate(cfg)
        self.experts = nn.ModuleList(
            Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit) for _ in range(cfg.n_routed_experts)
        )
        self.shared_experts = Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit)
        self.route_log: list | None = None  # set by Transformer.routing_trace()

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x, seq_len=shape[1] if len(shape) == 3 else None)
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        used = torch.unique(indices).tolist()
        if self.route_log is not None:
            self.route_log.append(tuple(used))
        # only experts that received tokens run, so unrouted experts keep grad=None this step
        for e in used:
            tok, slot = torch.where(indices == e)
            y = y.index_add(0, tok, self.experts[e](x[tok], weights[tok, slot, None]).float())
        y = y + self.shared_experts(x).float()
        return y.to(x.dtype).view(shape)


# --------------------------------------------------------------------------------------------------
# Engram n-gram memory
# --------------------------------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    return all(n % f for f in range(3, r + 1, 2))


def find_next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def engram_primes(cfg: Config) -> list[list[list[int]]]:
    """[layer][n-gram size - 2][head] bucket moduli: consecutive unused primes above engram_vocab_size."""
    seen, primes = set(), []
    for _ in cfg.engram_layer_ids:
        per_ngram = []
        for _ in range(cfg.engram_max_ngram_size - 1):
            sizes, current = [], cfg.engram_buckets - 1
            for _ in range(cfg.engram_n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return primes


def hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    """One odd multiplier per (layer, look-back), from a per-layer RNG (as the reference)."""
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    rows = [
        torch.tensor(
            np.random.default_rng(10007 * lid).integers(0, bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1
        )
        for lid in layer_ids
    ]
    return torch.stack(rows)


class NgramHash(nn.Module):
    """Hash ids of the 2..max_ngram_size-grams ending at each position, per Engram layer and head.

    The reference first maps tokenizer ids onto a normalized ("compressed") vocab.
    The race vocab is already tiny and canonical, so the map is the identity plus
    one extra id used to pad look-back before the start of the sequence.
    """

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.max_ngram = cfg.engram_max_ngram_size
        self.pad_id = vocab_size
        primes = engram_primes(cfg)
        offsets = [np.cumsum([0] + [p for per in layer for p in per][:-1]) for layer in primes]
        self.num_embeddings = [sum(p for per in layer for p in per) for layer in primes]
        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer(
            "multipliers", hash_multipliers(cfg.engram_layer_ids, self.max_ngram, vocab_size + 1), persistent=False
        )

    @torch.no_grad()
    def forward(self, input_ids):
        """[b, s] -> [b, s, n_engram_layers, (max_ngram - 1) * n_heads] row ids into each layer's table."""
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device)
        tokens = []
        for shift in range(self.max_ngram):
            src = input_ids.gather(1, (pos - shift).clamp_min(0).expand(b, s))
            tokens.append(torch.where((pos < shift).expand(b, s), self.pad_id, src))
        tokens = torch.stack(tokens, dim=-1)  # [b, s, max_ngram]
        products = tokens.unsqueeze(2) * self.multipliers  # [b, s, layers, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class Engram(nn.Module):
    """Adds an n-gram lookup to the residual stream, gated per hc copy by how well it matches it."""

    def __init__(self, cfg: Config, table_rows: int):
        super().__init__()
        self.dim, self.hc = cfg.dim, cfg.hc_mult
        self.eps, self.clamp_value = cfg.norm_eps, 1e-6
        n_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        self.embed = nn.Embedding(table_rows, cfg.engram_head_dim)
        self.wkv = nn.Linear(n_cols * cfg.engram_head_dim, cfg.dim * (cfg.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))
        self.k_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))

    def forward(self, h, hash_ids):
        """h [b, s, hc, dim]; hash_ids [b, s, n_cols]."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        hf = h.float()
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hf * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        return (hf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)


# --------------------------------------------------------------------------------------------------
# block and model
# --------------------------------------------------------------------------------------------------


class Block(nn.Module):
    """Attention + MoE on an ``hc_mult``-copy residual stream (manifold-constrained hyper-connections).

    Each sublayer derives (pre, post, comb) from the stream it reads; its ``pre``
    is used by the *next* sublayer to collapse the copies into one input (so the
    attention uses the previous block's FFN mix and the FFN uses this attention's).
    """

    def __init__(self, cfg: Config, layer_id: int, engram_rows: int | None):
        super().__init__()
        self.layer_id = layer_id
        self.hc, self.iters, self.hc_eps, self.norm_eps = cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps
        self.attn = Attention(cfg, layer_id)
        self.ffn = MoE(cfg)
        self.engram = Engram(cfg, engram_rows) if engram_rows is not None else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        mix_hc, hc_dim = (2 + self.hc) * self.hc, self.hc * cfg.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_mixes(self, x, fn, scale, base):
        with fp32_region(x):
            xf = x.flatten(2).float()
            mixes = F.linear(xf, fn.float()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
            return hc_split_sinkhorn(mixes, scale.float(), base.float(), self.hc, self.iters, self.hc_eps)

    @staticmethod
    def hc_pre(x, pre_mix):
        """Collapse the copies: sum_i pre_mix[i] * x[i]. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        with fp32_region(x):
            return torch.einsum("bsi,bsid->bsd", pre_mix, x.float()).to(x.dtype)

    @staticmethod
    def hc_post(x, residual, post, comb):
        """Expand back to copies and mix the residual in: y[j] = post[j] * x + sum_i comb[i, j] * residual[i]."""
        with fp32_region(x):
            y = post.unsqueeze(-1) * x.float().unsqueeze(-2) + torch.einsum("bsij,bsid->bsjd", comb, residual.float())
        return y.to(residual.dtype)  # the residual stream keeps its (fp32) dtype under bf16 autocast

    def forward(self, x, pre_mix, ctx: AttnContext):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), ctx)
        x = self.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)))
        x = self.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


@dataclass(frozen=True)
class ParamRole:
    """How an optimizer should treat a parameter (see ``Transformer.param_roles``).

    kind
        ``matrix``: a hidden linear weight (Muon territory; includes the router, mHC
        ``hc_*_fn``, Engram ``wkv`` and every expert). ``embedding`` / ``head``: token
        embedding and output head. ``engram_table``: hashed n-gram rows. ``vector``: norm
        gains, sinks, mHC static biases/scales and Engram's 2-D elementwise gains.
    blocks
        Rows are a stack of this many *independent* matrices (``wo_a`` holds one
        ``[o_lora_rank, heads_per_group * head_dim]`` matrix per output group). Matrix
        optimizers must never treat such a weight as one matrix.
    head_blocks
        Optional finer split into attention heads (``wq_b``, ``indexer.wq_b``): Kimi K3's
        Per-Head Muon orthogonalizes each head's block separately.
    lm_grad
        False for the sparse indexer: its hard top-k sends it no language-model gradient.
    decay
        Whether weight decay applies in DeepSeek's recipe (no decay on sinks, mHC static
        biases/scales and Engram tables; norms *are* decayed).
    lr_scale
        Learning-rate multiplier in DeepSeek's recipe (Engram tables train at 5x).
    """

    kind: str
    blocks: int = 1
    head_blocks: int = 1
    lm_grad: bool = True
    decay: bool = True
    lr_scale: float = 1.0


class Transformer(nn.Module):
    """embed -> expand to hc_mult copies -> [Engram] -> blocks -> collapse -> norm -> head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.hc = cfg.hc_mult
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.engram_hash = NgramHash(cfg, cfg.vocab_size) if cfg.engram_layer_ids else None
        rows = dict(zip(cfg.engram_layer_ids, self.engram_hash.num_embeddings)) if self.engram_hash else {}
        self.engram_index = {lid: i for i, lid in enumerate(cfg.engram_layer_ids)}
        self.layers = nn.ModuleList(Block(cfg, i, rows.get(i)) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.reset_parameters()

    # ---- initialization (not specified by the inference code) ----
    @torch.no_grad()
    def reset_parameters(self):
        std = self.cfg.init_std
        for name, p in self.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("hc_attn_fn", "hc_ffn_fn"):
                p.normal_(0.0, std)
            elif leaf in ("hc_attn_scale", "hc_ffn_scale"):
                p.fill_(0.01)  # mHC paper: gating factor init 0.01
            elif leaf in ("hc_attn_base", "hc_ffn_base"):
                # Static hyper-connection init read off the released weights (identical in all 80 hc
                # modules): pre selects copy layer_id % hc (+3 / -3 logits), post is 0 (2*sigmoid -> 1),
                # comb is +3 on the diagonal and -3 off it (near-identity after Sinkhorn).
                hc, layer_id = self.hc, int(name.split(".")[1])
                p[:hc] = -3.0
                p[layer_id % hc] = 3.0
                p[hc : 2 * hc] = 0.0
                comb = torch.full((hc, hc), -3.0)
                comb.fill_diagonal_(3.0)
                p[2 * hc :] = comb.flatten()
            elif leaf == "attn_sink":
                p.zero_()
            elif leaf in ("q_weight", "k_weight") or name.endswith("norm.weight"):
                p.fill_(1.0)
            elif p.ndim >= 2:
                p.normal_(0.0, std)
            else:
                p.zero_()

    # ---- forward ----
    def forward(self, input_ids, last_only: bool = True, return_aux: bool = False):
        """input_ids [b, s] -> logits [b, vocab] at the last position (default) or [b, s, vocab].

        With ``return_aux=True`` returns ``(logits, aux_loss)`` where ``aux_loss`` is the MoE
        sequence-wise balance loss (``balance_loss_alpha`` x sum over layers) of this forward,
        zero outside gradient-enabled training. Add it to the task loss.
        """
        b, s = input_ids.shape
        hashes = self.engram_hash(input_ids) if self.engram_hash is not None else None
        h = self.embed(input_ids).unsqueeze(2).expand(b, s, self.hc, -1).contiguous()
        pre_mix = torch.zeros(b, s, self.hc, device=h.device)
        pre_mix[..., 0] = 1.0
        ctx = AttnContext()
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, self.engram_index[layer.layer_id]])
            h, pre_mix = layer(h, pre_mix, ctx)
        h = Block.hc_pre(h, pre_mix)
        if last_only:
            h = h[:, -1]
        h = self.norm(h)
        with fp32_region(h):
            logits = F.linear(h.float(), self.head.weight.float())
        terms = [layer.ffn.gate.balance_loss for layer in self.layers if layer.ffn.gate.balance_loss is not None]
        for layer in self.layers:  # do not keep this forward's graph alive through the gates
            layer.ffn.gate.balance_loss = None
        if not return_aux:
            return logits
        aux = self.cfg.balance_loss_alpha * torch.stack(terms).sum() if terms else logits.new_zeros(())
        return logits, aux

    # ---- training utilities ----
    @contextmanager
    def routing_trace(self):
        """Collects, for every MoE forward inside the context, the experts that received tokens.

        The model's own FLOPs do not depend on routing, but an optimizer's can
        (work skipped for idle experts, or a meta step differentiating only
        through the experts its held-out batch reaches); the race keys its FLOP
        counts on this trace.
        """
        log: list = []
        moes = [m for m in self.modules() if isinstance(m, MoE)]
        for m in moes:
            m.route_log = log
        try:
            yield log
        finally:
            for m in moes:
                m.route_log = None

    @contextmanager
    def frozen_router_stats(self):
        """Forwards inside this context do not count toward the router's load statistics.

        Optimizers that run extra forwards (a sharpness-aware perturbed pass, a meta
        step) must not change the load-balancing update, which is meant to see each
        training step's batch once.
        """
        gates = [m for m in self.modules() if isinstance(m, Gate)]
        previous = [g.track_load for g in gates]
        for g in gates:
            g.track_load = False
        try:
            yield
        finally:
            for g, prev in zip(gates, previous):
                g.track_load = prev

    @torch.no_grad()
    def update_router_bias(self, rate: float | None = None):
        """Auxiliary-loss-free load balancing: nudge each expert's selection bias toward the mean load.

        Uses the token counts accumulated by every gradient-enabled training forward since the last call.
        """
        rate = self.cfg.router_bias_update_rate if rate is None else rate
        for m in self.modules():
            if isinstance(m, Gate) and m.load.sum() > 0:
                m.bias += rate * torch.sign(m.load.mean() - m.load)
                m.load.zero_()

    def param_roles(self) -> dict[str, ParamRole]:
        cfg, roles = self.cfg, {}
        for name, p in self.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            lm_grad = ".indexer." not in name
            if name == "embed.weight":
                role = ParamRole("embedding")
            elif name == "head.weight":
                role = ParamRole("head")
            elif ".engram.embed." in name:
                role = ParamRole("engram_table", decay=False, lr_scale=5.0)
            elif leaf in ("q_weight", "k_weight", "attn_sink") or "hc_" in leaf and not leaf.endswith("_fn"):
                role = ParamRole("vector", decay=leaf in ("q_weight", "k_weight"))
            elif p.ndim < 2:
                role = ParamRole("vector", lm_grad=lm_grad)
            elif name.endswith("attn.wq_b.weight"):
                role = ParamRole("matrix", head_blocks=cfg.n_heads)
            elif name.endswith("indexer.wq_b.weight"):
                role = ParamRole("matrix", head_blocks=cfg.index_n_heads, lm_grad=False)
            elif name.endswith("attn.wo_a.weight"):
                role = ParamRole("matrix", blocks=cfg.o_groups)
            else:
                role = ParamRole("matrix", lm_grad=lm_grad)
            roles[name] = role
        return roles

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_active_params(self) -> int:
        """Parameters touched per token: everything except unrouted experts and unread table rows."""
        cfg = self.cfg
        expert = 3 * cfg.dim * cfg.moe_inter_dim
        routed_total = cfg.n_layers * cfg.n_routed_experts * expert
        routed_active = cfg.n_layers * cfg.n_activated_experts * expert
        tables = sum(m.embed.weight.numel() for m in self.modules() if isinstance(m, Engram))
        engram_active = sum(
            ((cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads * cfg.engram_head_dim)
            for m in self.modules()
            if isinstance(m, Engram)
        )
        embed_inactive = self.embed.weight.numel() - cfg.dim
        return self.num_params() - routed_total + routed_active - tables + engram_active - embed_inactive

    def train_flops_per_token(self, seq_len: int) -> float:
        """Approximate training FLOPs per token: 6 x active matmul params + attention score/value products."""
        cfg = self.cfg
        matmul = self.num_active_params() - cfg.dim  # the one embedding row is a lookup, not a matmul
        attn = 0
        for layer in self.layers:
            n = min(seq_len, cfg.window_size)
            if layer.attn.ratio:
                n += min(seq_len // layer.attn.ratio, cfg.index_topk)
            attn += 2 * 2 * cfg.n_heads * cfg.head_dim * n  # QK^T and PV, per token
        return 6 * matmul + 3 * attn


def build_model(
    preset: str = "tiny", vocab_size: int | None = None, max_seq_len: int | None = None, **overrides
) -> Transformer:
    from .config import get_config

    if vocab_size is not None:
        overrides["vocab_size"] = vocab_size
    if max_seq_len is not None:
        overrides["max_seq_len"] = max(max_seq_len, 1)
    return Transformer(get_config(preset, **overrides))
