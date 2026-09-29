"""The port computes the same function as the official DeepSeek-V4.1-Flash inference code.

Loads ``tests/reference_deepseek/model.py`` (the unmodified official file) with pure-PyTorch
kernel stand-ins, copies the port's weights into it, and compares last-position logits over
many sequence lengths with a short window and small top-k, so sliding-window masking,
compression, re-indexing with candidate blocks, Engram, MoE routing and mHC are all live.
"""

import importlib
import os
import sys
from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F

from deepseek_v41 import Transformer, get_config

REF_DIR = os.path.join(os.path.dirname(__file__), "reference_deepseek")
STUBS = ("model", "engram", "kernel", "vision", "image_processor", "sympy")


@pytest.fixture(scope="module")
def ref():
    saved = {name: sys.modules.pop(name) for name in STUBS if name in sys.modules}
    sys.path.insert(0, REF_DIR)
    try:
        engram = importlib.import_module("engram")
        model = importlib.import_module("model")
        yield model, engram
    finally:
        sys.path.remove(REF_DIR)
        for name in STUBS:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def port_config(vocab):
    return get_config(
        "tiny",
        vocab_size=vocab,
        window_size=4,
        index_topk=3,
        index_n_heads=8,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_head_dim=32,
        max_seq_len=64,
        original_seq_len=16,
        rope_factor=4.0,
    )


def build_reference(model_mod, engram_mod, cfg, port, vocab, batch):
    # identity "compressed tokenizer" with one extra id as the look-back pad, as in the port
    engram_mod.build_compressed_token_map = lambda tok: (list(range(vocab + 1)), vocab + 1)
    names = {f.name for f in fields(model_mod.ModelArgs)}
    args = model_mod.ModelArgs(**{k: v for k, v in cfg.to_dict().items() if k in names})
    args.engram_vocab_size = cfg.engram_buckets
    args.max_batch_size, args.dtype, args.expert_dtype, args.n_mtp_layers = batch, "bf16", None, 0
    args.engram_num_embeddings = tuple(port.engram_hash.num_embeddings)
    args.engram_compressed_vocab_size, args.engram_pad_id = vocab + 1, vocab
    ref = model_mod.Transformer(args, tokenizer=None).float()

    def plain_table_lookup(self, indices):  # the fp8 table with unit scales is a plain lookup
        return F.embedding(indices, self.weight)

    for m in ref.modules():
        if type(m).__name__ == "ParallelEngramEmbedding":
            m.forward = plain_table_lookup.__get__(m)
    missing, unexpected = ref.load_state_dict(port.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(k.endswith(".scale") for k in missing), missing
    return ref


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5, 8, 11, 17])
def test_logits_match_official_reference(ref, seq_len):
    model_mod, engram_mod = ref
    vocab, batch = 61, 3
    torch.manual_seed(0)
    cfg = port_config(vocab)
    port = Transformer(cfg)
    with torch.no_grad():  # push mHC and the router away from their symmetric init so every path matters
        for n, p in port.named_parameters():
            if "hc_" in n and ("base" in n or "scale" in n):
                p.add_(torch.randn_like(p) * 0.5)
            if n.endswith("attn_sink"):
                p.normal_()
    for m in port.modules():
        if type(m).__name__ == "Gate":
            m.bias.normal_(0, 0.05)
    reference = build_reference(model_mod, engram_mod, cfg, port, vocab, batch)
    x = torch.randint(0, vocab, (batch, seq_len))
    with torch.no_grad():
        mine = port(x)
    _, theirs, _ = reference(x)
    assert torch.allclose(mine, theirs, atol=1e-5, rtol=0), (mine - theirs).abs().max()
