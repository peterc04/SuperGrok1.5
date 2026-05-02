"""Reference implementation of the Decoder Transformer.

Verbatim copy of ``DecoderBlock`` and ``Transformer`` from
``grokking_race_v2.py`` lines 318-340 (Model 1: Decoder Transformer).
Only imports ``torch`` + ``torch.nn`` so it is safe to import from a
test fixture without pulling in the full training script.

The reference is the ground-truth oracle for parity tests against the
fused kernel binding ``_ops.models.decoder_forward``. Do NOT refactor
or "optimize" this code — it must match the upstream definition byte
for byte. Lint-only changes (whitespace, blank lines for class
boundaries) are acceptable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Model 1: Decoder Transformer ─────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, d, h, seq_len=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.register_buffer('causal_mask', torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1))
    def forward(self, x):
        a, _ = self.attn(x, x, x, attn_mask=self.causal_mask)
        x = self.n1(x + a); return self.n2(x + self.ff(x))


class Transformer(nn.Module):
    def __init__(self, nl=2, d=128, h=4, ntok=99, seq=4):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(seq, d)
        self.layers = nn.ModuleList([DecoderBlock(d, h, seq_len=seq) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, ntok)
        self.register_buffer('pos_ids', torch.arange(seq).unsqueeze(0))
    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        for l in self.layers: h = l(h)
        return self.out(self.norm(h)[:, -1, :])


def make_decoder(nl: int = 2, d: int = 128, h: int = 4,
                 ntok: int = 99, seq: int = 4) -> Transformer:
    """Construct a fresh ``Transformer`` with the parity-test default config.

    Defaults match the small-model parity test parameters in
    ``tests/test_models_sm_90.py``: ``nl=2, d=128, h=4, ntok=99, seq=4``.
    """
    return Transformer(nl=nl, d=d, h=h, ntok=ntok, seq=seq)
