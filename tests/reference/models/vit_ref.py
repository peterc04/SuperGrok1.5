"""Reference implementation of the Vision Transformer (ViT).

Verbatim copy of ``EncoderBlock`` and ``ViT`` from
``grokking_race_v2.py`` lines 342-368 (Model 2: ViT). Only imports
``torch`` + ``torch.nn`` so it is safe to import from a test fixture
without pulling in the full training script.

The reference is the ground-truth oracle for parity tests against the
fused kernel binding ``_ops.models.vit_forward``. Do NOT refactor or
"optimize" this code — it must match the upstream definition byte for
byte. Lint-only changes (whitespace, blank lines for class
boundaries) are acceptable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Model 2: ViT ─────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a); return self.n2(x + self.ff(x))


class ViT(nn.Module):
    def __init__(self, p=97, patch_dim=49, num_patches=16, d=128, h=4, nl=2):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Embedding(num_patches + 1, d)
        self.layers = nn.ModuleList([EncoderBlock(d, h) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, p)
        self.register_buffer('pos_ids', torch.arange(num_patches + 1).unsqueeze(0))
    def forward(self, x):
        B = x.size(0); h = self.patch_proj(x)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = h + self.pos(self.pos_ids)
        for l in self.layers: h = l(h)
        return self.out(self.norm(h[:, 0, :]))


def make_vit(p: int = 97, patch_dim: int = 49, num_patches: int = 16,
             d: int = 128, h: int = 4, nl: int = 2) -> ViT:
    """Construct a fresh ``ViT`` with the parity-test default config.

    Defaults match the small-model parity test parameters in
    ``tests/test_models_sm_90.py``: ``p=97, patch_dim=49,
    num_patches=16, d=128, h=4, nl=2``.
    """
    return ViT(p=p, patch_dim=patch_dim, num_patches=num_patches,
               d=d, h=h, nl=nl)
