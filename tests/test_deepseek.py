"""DeepSeek-V4.1-Flash port: shapes, causality, the attention/Sinkhorn/router/Engram pieces, parameter counts."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from deepseek_v41 import Transformer, build_model, get_config
from deepseek_v41.model import (
    Gate,
    NgramHash,
    engram_primes,
    hash_multipliers,
    hc_split_sinkhorn,
    select_candidate_blocks,
    sink_attention,
)


def small_window_config(**kw):
    """Tiny, but with a short window and small top-k so masking and selection actually bite."""
    base = dict(window_size=4, index_topk=3, candidate_topk_blocks=2, candidate_block_size=2, max_seq_len=64)
    base.update(kw)
    return get_config("tiny", **base)


# ---------------------------------------------------------------- building blocks


def test_sink_attention_matches_gather_reference():
    """Dense masked attention == the reference kernel's gather-by-index semantics (with -1 = empty slot)."""
    g = torch.Generator().manual_seed(0)
    b, s, h, d, n, k = 2, 5, 3, 8, 9, 4
    q, kv, sink = torch.randn(b, s, h, d, generator=g), torch.randn(b, n, d, generator=g), torch.randn(h, generator=g)
    idx = torch.randint(-1, n, (b, s, k), generator=g)
    idx[0, 0] = -1  # a query with nothing to attend to
    mask = torch.zeros(b, s, n, dtype=torch.bool)
    for bi in range(b):
        for si in range(s):
            for j in idx[bi, si]:
                if j >= 0:
                    mask[bi, si, j] = True
    out = sink_attention(q, kv, mask, sink, d**-0.5)
    ref = torch.zeros_like(q)
    for bi in range(b):
        for si in range(s):
            sel = sorted({int(j) for j in idx[bi, si] if j >= 0})
            for hi in range(h):
                if not sel:
                    continue
                scores = torch.stack([q[bi, si, hi] @ kv[bi, j] for j in sel]) * d**-0.5
                m = scores.max()
                w = torch.exp(scores - m)
                ref[bi, si, hi] = (w[:, None] * kv[bi, sel]).sum(0) / (w.sum() + torch.exp(sink[hi] - m))
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.equal(out[0, 0], torch.zeros_like(out[0, 0]))


def test_sinkhorn_comb_is_doubly_stochastic():
    g = torch.Generator().manual_seed(0)
    hc = 4
    mixes = torch.randn(3, 5, (2 + hc) * hc, generator=g)
    pre, post, comb = hc_split_sinkhorn(mixes, torch.ones(3), torch.zeros((2 + hc) * hc), hc, 20, 1e-6)
    # the last Sinkhorn step normalizes columns exactly; 20 iterations leave rows close (as in the reference)
    assert torch.allclose(comb.sum(-2), torch.ones(3, 5, hc), atol=1e-5)
    assert torch.allclose(comb.sum(-1), torch.ones(3, 5, hc), atol=1e-2)
    assert (pre > 0).all() and (pre < 1 + 1e-5).all() and (post > 0).all() and (post < 2).all()


def test_router_bias_selects_but_does_not_weight():
    cfg = get_config("tiny")
    gate = Gate(cfg)
    torch.nn.init.normal_(gate.weight, std=0.1)
    x = torch.randn(16, cfg.dim)
    w0, i0 = gate(x)
    gate.bias[:] = 0
    gate.bias[3] = 100.0  # force expert 3 into every token's top-k
    w1, i1 = gate(x)
    assert (i1 == 3).any(dim=-1).all()
    # weights are the (normalized, scaled) unbiased scores of whichever experts were chosen
    scores = F.softplus(x @ gate.weight.T).sqrt()
    ref = scores.gather(1, i1)
    ref = ref / ref.sum(-1, keepdim=True) * cfg.route_scale
    assert torch.allclose(w1, ref, atol=1e-6)


def test_router_bias_update_moves_toward_balance():
    model = build_model("tiny", vocab_size=20, max_seq_len=8)
    gate = model.layers[0].ffn.gate
    gate.load[:] = torch.arange(gate.load.numel(), dtype=torch.float32)
    model.update_router_bias(rate=0.01)
    mean = (gate.load.numel() - 1) / 2
    expect = 0.01 * torch.sign(mean - torch.arange(gate.load.numel(), dtype=torch.float32))
    assert torch.allclose(gate.bias, expect)
    assert gate.load.sum() == 0


def test_candidate_blocks_keep_newest_block():
    scores = torch.tensor([[[5.0, 4.0, -1.0, -2.0, 0.0, float("-inf")]]])
    keep = select_candidate_blocks(scores, torch.tensor([[5]]), topk_blocks=1, block_size=2)
    # 1 block allowed, and the block holding the newest visible position (index 4 -> block 2) is pinned
    assert keep.tolist() == [[[False, False, False, False, True, True]]]


# ---------------------------------------------------------------- Engram hashing


def naive_ngram_hash(ids, cfg, vocab):
    """Direct transcription of the reference NgramHashState.forward for one sequence (identity token map)."""
    primes = engram_primes(cfg)
    mult = hash_multipliers(cfg.engram_layer_ids, cfg.engram_max_ngram_size, vocab + 1).tolist()
    out = []
    for pos in range(len(ids)):
        per_layer = []
        for li, _ in enumerate(cfg.engram_layer_ids):
            toks = [ids[pos - sh] if pos - sh >= 0 else vocab for sh in range(cfg.engram_max_ngram_size)]
            flat = [p for per in primes[li] for p in per]
            offsets = np.cumsum([0] + flat[:-1]).tolist()
            rolling, cols = toks[0] * mult[li][0], []
            for i in range(1, cfg.engram_max_ngram_size):
                rolling ^= toks[i] * mult[li][i]
                cols += [rolling % p for p in primes[li][i - 1]]
            per_layer.append([c + o for c, o in zip(cols, offsets)])
        out.append(per_layer)
    return torch.tensor(out)


def test_engram_hash_matches_reference_transcription_and_bounds():
    cfg, vocab = get_config("tiny"), 99
    hasher = NgramHash(cfg, vocab)
    ids = torch.tensor([[3, 97, 5, 98, 7, 97, 11, 98]])
    got = hasher(ids)[0]
    assert torch.equal(got, naive_ngram_hash(ids[0].tolist(), cfg, vocab))
    for li, rows in enumerate(hasher.num_embeddings):
        assert int(got[:, li].max()) < rows and int(got[:, li].min()) >= 0
    assert torch.equal(hasher(ids), got.unsqueeze(0))  # deterministic


# ---------------------------------------------------------------- whole model


def test_shapes_and_last_only():
    model = build_model("tiny", vocab_size=99, max_seq_len=8)
    x = torch.randint(0, 99, (3, 8))
    full = model(x, last_only=False)
    last = model(x)
    assert full.shape == (3, 8, 99) and last.shape == (3, 99)
    assert torch.allclose(full[:, -1], last, atol=1e-5)


@pytest.mark.parametrize("seq_len", [1, 2, 3, 5, 9, 23])
def test_causal_no_future_leak(seq_len):
    """Changing token t must not change logits at positions < t (window, compression, indexer, Engram)."""
    model = Transformer(small_window_config(vocab_size=50)).eval()
    x = torch.randint(0, 50, (2, seq_len))
    ref = model(x, last_only=False)
    for t in range(seq_len):
        x2 = x.clone()
        x2[:, t] = (x2[:, t] + 7) % 50
        out = model(x2, last_only=False)
        assert torch.allclose(out[:, :t], ref[:, :t], atol=1e-5), f"leak into positions < {t}"


def test_gradients_reach_every_trained_parameter():
    torch.manual_seed(0)
    model = Transformer(small_window_config(vocab_size=50))
    x = torch.randint(0, 50, (64, 12))
    loss = F.cross_entropy(model(x, last_only=False).flatten(0, 1), torch.randint(0, 50, (64 * 12,)))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None and ".indexer." not in n]
    assert not missing, missing
    # the indexer makes a hard top-k choice, so the LM loss sends it no gradient
    assert all(p.grad is None for n, p in model.named_parameters() if ".indexer." in n)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_same_seed_same_init():
    torch.manual_seed(7)
    a = build_model("tiny", vocab_size=99, max_seq_len=8)
    torch.manual_seed(7)
    b = build_model("tiny", vocab_size=99, max_seq_len=8)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_router_load_is_tracked_only_in_training():
    model = build_model("tiny", vocab_size=99, max_seq_len=8)
    x = torch.randint(0, 99, (8, 4))
    with torch.no_grad():
        model(x)
    assert all(m.load.sum() == 0 for m in model.modules() if isinstance(m, Gate))
    model(x).sum().backward()
    gate = model.layers[0].ffn.gate
    assert gate.load.sum() == 8 * 4 * get_config("tiny").n_activated_experts


def test_param_roles_cover_everything():
    model = build_model("tiny", vocab_size=99, max_seq_len=8)
    roles = model.param_roles()
    assert set(roles) == {n for n, _ in model.named_parameters()}
    assert roles["layers.0.attn.wq_b.weight"].head_blocks == get_config("tiny").n_heads
    assert roles["layers.0.attn.wo_a.weight"].blocks == get_config("tiny").o_groups
    assert roles["layers.1.engram.q_weight"].kind == "vector"  # 2-D, but an elementwise gain
    assert not roles["layers.1.attn.indexer.wq_b.weight"].lm_grad
    assert not roles["layers.0.hc_attn_base"].decay and roles["layers.1.engram.embed.weight"].lr_scale == 5.0
    assert roles["embed.weight"].kind == "embedding" and roles["head.weight"].kind == "head"
    assert roles["layers.1.engram.embed.weight"].kind == "engram_table"
    # every expert, the router and mHC's dynamic projection are hidden matrices (DeepSeek-V4 trains them with Muon)
    for name in ("layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.gate.weight", "layers.0.hc_attn_fn"):
        assert roles[name].kind == "matrix", name
    # the router bias is a buffer, never an optimizer parameter
    assert "layers.0.ffn.gate.bias" not in roles


def test_flash_preset_matches_published_parameter_counts():
    with torch.device("meta"):
        model = Transformer(get_config("flash"))
    engram = sum(p.numel() for n, p in model.named_parameters() if ".engram.embed." in n)
    backbone = model.num_params() - engram
    assert abs(backbone / 1e9 - 552) < 1.0  # model card: 552B backbone
    assert abs(engram / 1e9 - 196) < 1.0  # model card: 196B Engram


def test_engram_tables_follow_the_final_vocab_and_h100_is_7_to_10b():
    """Regression: the Engram bucket count used to be resolved from the preset's 129K vocab before the
    race's 99-token vocab was applied, silently adding 63B parameters of tables to the h100 preset."""
    assert build_model("tiny", vocab_size=99, max_seq_len=4).cfg.engram_buckets == 160 * 100
    with torch.device("meta"):
        model = Transformer(get_config("h100", vocab_size=99))
    assert 7e9 < model.num_params() < 10e9
    assert 1.0e9 < model.num_active_params() < 2.0e9
