"""Reference implementation of the Mamba selective-state-space model.

Verbatim copy of ``SelectiveSSMLayer`` and ``MambaModel`` from
``grokking_race_v2.py`` lines 369-424 (Model 3: Mamba SSM). Only
imports ``torch`` + ``torch.nn`` + ``torch.nn.functional`` so it is
safe to import from a test fixture without pulling in the full
training script.

The reference is the ground-truth oracle for parity tests against the
fused kernel binding ``_ops.models.mamba_forward`` (and the per-layer
binding ``_ops.models.mamba_layer_forward``). Do NOT refactor or
"optimize" this code — it must match the upstream definition byte for
byte. Lint-only changes (whitespace, blank lines for class
boundaries) are acceptable.

Note: ``SelectiveSSMLayer._selective_scan`` will fall back to the
Python scan loop if ``mamba_scan_ext`` is not importable. That is the
intended behaviour from the upstream definition and matches what the
parity-test oracle should run on a CPU-only host.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model 3: Mamba SSM ───────────────────────────────────────────────
class SelectiveSSMLayer(nn.Module):
    def __init__(self, d, state_dim=16, dt_rank=None, expand_factor=2):
        super().__init__()
        self.state_dim = state_dim; self.d_inner = d * expand_factor
        self.dt_rank = dt_rank or max(d // 16, 1)
        self.in_proj = nn.Linear(d, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3,
                                padding=1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)
        self.norm = nn.LayerNorm(d)
    def _selective_scan(self, x, dt, B, C):
        batch, L, _ = x.shape; A = -torch.exp(self.A_log); dt = F.softplus(dt)
        # Try CUDA kernel
        if x.is_cuda:
            try:
                from mamba_scan_ext import selective_scan_cuda
                return selective_scan_cuda(
                    x.contiguous(), dt.contiguous(),
                    B.contiguous(), C.contiguous(), A.contiguous()
                )
            except ImportError:
                pass
        # Python fallback
        h = torch.zeros(batch, self.d_inner, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            h = torch.exp(dt_t * A.unsqueeze(0)) * h + (dt_t * B[:, t, :].unsqueeze(1)) * x[:, t, :].unsqueeze(-1)
            ys.append((h * C[:, t, :].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1)
    def forward(self, x):
        residual = x; xz = self.in_proj(x); x_main, z = xz.chunk(2, dim=-1)
        x_main = F.silu(self.conv1d(x_main.transpose(1, 2)).transpose(1, 2))
        x_dbc = self.x_proj(x_main)
        dt, B, C = x_dbc.split([self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        y = self._selective_scan(x_main, self.dt_proj(dt), B, C)
        y = self.out_proj((y + x_main * self.D.unsqueeze(0).unsqueeze(0)) * F.silu(z))
        return self.norm(y + residual)


class MambaModel(nn.Module):
    def __init__(self, p=97, ntok=99, seq_len=8, d=128, nl=2):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(seq_len, d)
        self.layers = nn.ModuleList([SelectiveSSMLayer(d) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, p)
        self.register_buffer('pos_ids', torch.arange(seq_len).unsqueeze(0))
    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        for l in self.layers: h = l(h)
        return self.out(self.norm(h[:, -1, :]))


def make_mamba(p: int = 97, ntok: int = 99, seq_len: int = 8,
               d: int = 128, nl: int = 2) -> MambaModel:
    """Construct a fresh ``MambaModel`` with the parity-test default config.

    Defaults match the small-model parity test parameters in
    ``tests/test_models_sm_90.py``: ``p=97, ntok=99, seq_len=8, d=128,
    nl=2``.
    """
    return MambaModel(p=p, ntok=ntok, seq_len=seq_len, d=d, nl=nl)
