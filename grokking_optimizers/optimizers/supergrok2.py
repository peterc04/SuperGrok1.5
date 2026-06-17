"""
SuperGrok v2 — CSA/HCA Hybrid Attention + 4-Head PEER + GRU Meta-Net Optimizer

Replaces the previous Mamba-3 bidirectional selective-scan SEQUENCE MIXER with a
DeepSeek-V4-style CSA/HCA hybrid compressed attention stack. The GRU + PEER
product-key routing + per-element expert MLP + Adam apply tail are kept verbatim
(per spec §3b — the tail is independent of the sequence mixer).

The meta-model is sequence→sequence (N gradient elements → N smart values). The
two attention contexts replace the two scan outputs one-for-one (same shapes):

  - CSA (Compressed Sparse Attention, m=4, sliding window=8, top-k):
      compresses KV by strided weighted pooling, uses a low-rank "lightning
      indexer" to pick top-k compressed entries per query, plus a causal sliding
      window of raw tokens, then multi-head softmax attention. Produces
      ``csa_ctx`` — the fine-grained/local context (was ``mamba_fwd`` output).
  - HCA (Heavily Compressed Attention, m'=128, dense):
      mean-pools KV at stride 128, dense attention over ALL compressed entries
      plus the sliding window. Produces ``hca_ctx`` — the global coarse context
      (was ``mamba_bwd`` output).

Key features:
  - CSA + HCA hybrid attention sequence mixer (sorted by |gradient| magnitude)
  - 4-Head PEER product-key expert routing (144 experts, 4 active per element)
  - Per-element GRU for temporal gradient memory (carried across steps)
  - Dynamic expert recycling (dead experts cloned from top performer)
  - All adaptive scheduling from v1.5 (sigmoid SAM/bilevel/WD, alpha updates)
  - functional_call SAM (no parameter modification)
  - CUDA-only execution path (no Python reference fallback)

Notes:
  - Attention is STATELESS across optimizer steps (unlike Mamba's carried scan
    state). The GRU state is still carried across steps.
  - All meta-model accumulation is FP32.
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint as _grad_checkpoint
from typing import Optional, Dict, List, Tuple


# Activation/gradient checkpointing helper (mirrors grokking_race_v2.py).
# Only checkpoints when training AND grad is being tracked — under
# torch.no_grad()/eval there is nothing to recompute, so we call directly,
# avoiding checkpoint's overhead and its "no input requires grad" warning.
# This keeps the eval/inference and CUDA fused-step paths completely unaffected.
def _maybe_checkpoint(module, x, enabled):
    """Run ``module(x)`` (single-tensor input), optionally checkpointed."""
    if enabled and module.training and torch.is_grad_enabled():
        return _grad_checkpoint(module, x, use_reentrant=False)
    return module(x)


def _maybe_checkpoint_fn(fn, enabled, training, *args):
    """Run ``fn(*args)`` for an arbitrary self-contained sub-computation,
    optionally under activation checkpointing.

    Used for the in-forward sub-blocks that are not standalone nn.Modules
    (per-stage attention closures, PEER expert MLP). ``training`` is the
    owning module's ``self.training`` flag so eval bypasses recomputation.
    """
    if enabled and training and torch.is_grad_enabled():
        return _grad_checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)

from grokking_optimizers.dispatch import get_ops
from grokking_optimizers.dispatch import (
    get_device_sm, get_gpu_vendor,
    supports_bf16, supports_fp8, supports_tf32,
)

_ops = get_ops()  # Fails loudly if C++ extension not built
# The C++ pybind registers BOTH the new CSA/HCA names and the old mamba_peer
# aliases (same underlying fn) for back-compat. Detect either.
_HAS_CUDA = (hasattr(_ops, 'supergrok2_batched_step') or
             hasattr(_ops, 'supergrok2_mamba_peer_batched_step'))
_HAS_CUDA_BACKWARD = hasattr(_ops, 'supergrok2_bilevel_fwd_save')


def _ops_fn(new_name, old_name):
    """Prefer the new CSA/HCA pybind name; fall back to the old mamba name."""
    fn = getattr(_ops, new_name, None)
    if fn is None:
        fn = getattr(_ops, old_name)
    return fn


# ════════════════════════════════════════════════════════════════════════════
#  Precision configuration (formerly grokking_optimizers/_quantization.py)
# ════════════════════════════════════════════════════════════════════════════


class PrecisionConfig:
    """Configures precision for different components of the optimizer.

    Supported precision modes (3-arch active set):
      Projections: fp32, tf32, bf16, fp8
      Expert weights: fp32, int8, int4
      Scan state: always fp32 (numerical necessity)
    """

    PROJECTION_MODES = ('fp32', 'tf32', 'bf16', 'fp8', 'auto')
    EXPERT_MODES = ('fp32', 'int8', 'int4', 'auto')

    def __init__(
        self,
        projection_precision='auto',
        expert_precision='fp32',
        scan_precision='fp32',
        dynamic=False,
    ):
        self.scan_precision = 'fp32'
        self.dynamic = dynamic

        if projection_precision == 'auto':
            # 'auto' picks the live default: BF16 everywhere it is supported
            # (the configured default for SG2 — see SuperGrok2.__init__), only
            # dropping to FP32 on a host with no BF16 path. FP8/INT4 are NOT
            # selected by 'auto'; they remain explicitly opt-in (config-gated)
            # because they need the matching kernel ABI to be the live tail.
            vendor = get_gpu_vendor()  # noqa: F841 — kept for clarity/symmetry
            self.projection_precision = 'bf16' if supports_bf16() else 'fp32'
        else:
            if projection_precision not in self.PROJECTION_MODES:
                raise ValueError(
                    f"Unknown projection precision: {projection_precision}. "
                    f"Must be one of {self.PROJECTION_MODES}"
                )
            self.projection_precision = projection_precision

        if expert_precision == 'auto':
            self.expert_precision = 'int8'
        else:
            if expert_precision not in self.EXPERT_MODES:
                raise ValueError(
                    f"Unknown expert precision: {expert_precision}. "
                    f"Must be one of {self.EXPERT_MODES}"
                )
            self.expert_precision = expert_precision

        self._step_count = 0
        self._grad_norm_ema = None
        self._grad_norm_var_ema = None
        self._precision_tier = 0

    def convert_projection_weights(self, w):
        """Convert a projection weight matrix to the target precision."""
        if self.projection_precision in ('fp32', 'tf32'):
            return w.float().contiguous(), None
        elif self.projection_precision == 'bf16':
            if not supports_bf16():
                return w.float().contiguous(), None
            return w.bfloat16().contiguous(), None
        elif self.projection_precision == 'fp8':
            if not supports_fp8():
                if supports_bf16():
                    return w.bfloat16().contiguous(), None
                return w.float().contiguous(), None
            scale = w.abs().max().clamp(min=1e-12)
            w_scaled = w.float().div(scale)
            w_fp8 = w_scaled.to(torch.float8_e4m3fn)
            return w_fp8, scale
        else:
            raise ValueError(f"Unknown precision: {self.projection_precision}")

    def convert_expert_weights(self, w1, b1, w2, b2):
        """Convert expert MLP weights to target precision."""
        if self.expert_precision == 'fp32':
            return {
                'w1': w1.float().contiguous(),
                'b1': b1.float().contiguous(),
                'w2': w2.float().contiguous(),
                'b2': b2.float().contiguous(),
                'mode': 'fp32',
            }
        elif self.expert_precision == 'int8':
            return self._quantize_expert_int8(w1, b1, w2, b2)
        elif self.expert_precision == 'int4':
            return self._quantize_expert_int4(w1, b1, w2, b2)
        else:
            raise ValueError(f"Unknown expert precision: {self.expert_precision}")

    def _quantize_expert_int8(self, w1, b1, w2, b2):
        def sym_quant(w):
            absmax = w.abs().max().clamp(min=1e-12)
            scale = absmax / 127.0
            q = (w / scale).round().clamp(-127, 127).to(torch.int8)
            return q, scale

        w1_q, w1_s = sym_quant(w1.float())
        w2_q, w2_s = sym_quant(w2.float())
        return {
            'w1_q': w1_q.contiguous(), 'w1_s': w1_s,
            'b1': b1.float().contiguous(),
            'w2_q': w2_q.contiguous(), 'w2_s': w2_s,
            'b2': b2.float().contiguous(),
            'mode': 'int8',
        }

    def _quantize_expert_int4(self, w1, b1, w2, b2):
        def int4_quant(w, group_size=32):
            w_flat = w.reshape(-1).float()
            N = w_flat.numel()
            N_padded = ((N + 1) // 2) * 2
            if N_padded > N:
                w_flat = torch.nn.functional.pad(w_flat, (0, N_padded - N))

            num_groups = (N_padded + group_size - 1) // group_size
            actual_gs = N_padded // num_groups
            w_grouped = w_flat.reshape(num_groups, actual_gs)

            gmax = w_grouped.max(dim=1).values
            gmin = w_grouped.min(dim=1).values
            scales = ((gmax - gmin) / 15.0).clamp(min=1e-12)
            zeros = gmin

            q = ((w_grouped - zeros.unsqueeze(1)) / scales.unsqueeze(1))
            q = q.round().clamp(0, 15).to(torch.uint8).reshape(-1)

            even = q[0::2]
            odd = q[1::2]
            packed = even | (odd << 4)
            return packed, scales, zeros

        w1_p, w1_s, w1_z = int4_quant(w1)
        w2_p, w2_s, w2_z = int4_quant(w2)
        return {
            'w1_packed': w1_p.contiguous(), 'w1_scales': w1_s, 'w1_zeros': w1_z,
            'b1': b1.float().contiguous(),
            'w2_packed': w2_p.contiguous(), 'w2_scales': w2_s, 'w2_zeros': w2_z,
            'b2': b2.float().contiguous(),
            'mode': 'int4',
        }

    def update_dynamic(self, grad_norm):
        """Update dynamic precision state based on gradient norm stability."""
        if not self.dynamic:
            return False

        self._step_count += 1
        alpha = 0.01

        if self._grad_norm_ema is None:
            self._grad_norm_ema = grad_norm
            self._grad_norm_var_ema = 0.0
            return False

        self._grad_norm_ema = (1 - alpha) * self._grad_norm_ema + alpha * grad_norm
        deviation = (grad_norm - self._grad_norm_ema) ** 2
        self._grad_norm_var_ema = (1 - alpha) * self._grad_norm_var_ema + alpha * deviation

        cv = math.sqrt(self._grad_norm_var_ema) / max(self._grad_norm_ema, 1e-12)

        if self._step_count < 500:
            return False

        changed = False
        if cv < 0.05 and self._precision_tier < 3:
            self._precision_tier = 3
            changed = True
        elif cv < 0.10 and self._precision_tier < 2:
            self._precision_tier = 2
            changed = True
        elif cv < 0.20 and self._precision_tier < 1:
            self._precision_tier = 1
            changed = True
        elif cv > 0.30 and self._precision_tier > 0:
            self._precision_tier = 0
            changed = True

        if changed:
            self._apply_dynamic_tier()
        return changed

    def _apply_dynamic_tier(self):
        vendor = get_gpu_vendor()
        if vendor == 'nvidia':
            proj_tiers = ['fp32', 'tf32', 'bf16', 'fp8']
        else:
            proj_tiers = ['fp32', 'fp32', 'bf16', 'bf16']
        expert_tiers = ['fp32', 'fp32', 'int8', 'int4']

        tier = min(self._precision_tier, len(proj_tiers) - 1)
        new_proj = proj_tiers[tier]
        if new_proj == 'fp8' and not supports_fp8():
            new_proj = 'bf16'
        if new_proj == 'bf16' and not supports_bf16():
            new_proj = 'fp32'
        if new_proj == 'tf32' and not supports_tf32():
            new_proj = 'fp32'
        self.projection_precision = new_proj
        self.expert_precision = expert_tiers[min(tier, len(expert_tiers) - 1)]

    @property
    def stability_cv(self):
        if self._grad_norm_ema is None or self._grad_norm_var_ema is None:
            return float('inf')
        return math.sqrt(self._grad_norm_var_ema) / max(self._grad_norm_ema, 1e-12)

    def __repr__(self):
        parts = [
            f"projection={self.projection_precision}",
            f"expert={self.expert_precision}",
            f"scan={self.scan_precision}",
        ]
        if self.dynamic:
            parts.append(f"dynamic=True (tier={self._precision_tier})")
        return f"PrecisionConfig({', '.join(parts)})"


# ════════════════════════════════════════════════════════════════════════════
#  CSA/HCA Hybrid Attention + 4-Head PEER + GRU meta-network
#  (formerly grokking_optimizers/_metanet.py)
#
#  Architecture per optimizer step, per parameter:
#    1. SORT by |gradient|
#    2. CSA + HCA HYBRID ATTENTION — cross-element awareness via compressed
#       sparse (local/fine) + heavily-compressed dense (global/coarse) attention
#    3. PER-ELEMENT GRU — temporal memory across optimizer steps
#    4. 4-HEAD PEER ROUTING — product-key lookup, 4 experts per element
#    5. EXPERT MLP — 144 experts, hidden=16 default
#    6. DYNAMIC EXPERT RECYCLING — dead experts cloned from top performer
#    7. SKIP CONNECTION — smart_grad = grad + rescale * sum(expert_outputs)
# ════════════════════════════════════════════════════════════════════════════


class HybridCompressedAttention(nn.Module):
    """DeepSeek-V4-style compressed attention sequence mixer.

    A single block instantiated twice (``mode='csa'`` and ``mode='hca'``) to
    produce the two context streams that replace the bidirectional Mamba scan.

    mode='csa' — Compressed Sparse Attention (compression ``csa_compress``=4):
      1. KV compression: each compressed entry pools a window of ``csa_window``=8
         tokens at stride ``csa_compress``=4 with a learned per-position weight
         (softmax over ``csa_compress_w``).  Compressed length Nc = ceil(N/4).
      2. Lightning indexer (top-k): a low-rank query ``qI = x @ idx_DQ @ idx_UQ``
         (rank ``indexer_rank``=4) scores each query token against compressed
         indexer keys ``kI`` (pooled ``x @ idx_K``). Keep the top-k
         (``csa_topk``=16, clamped to Nc) highest-scoring compressed entries.
      3. Sliding window: additionally attend to the last ``csa_window`` raw
         tokens (causal local context).
      4. Attention: multi-head softmax(Q·Kᵀ/sqrt(head_dim))·V over the union of
         the selected compressed entries and the window. Output → ``out_W``.

    mode='hca' — Heavily Compressed Attention (compression ``hca_compress``=128):
      1. KV compression: mean pool of ``hca_compress`` tokens at stride 128.
         Compressed length Nh = ceil(N/128).
      2. Dense attention: every query attends to ALL Nh compressed entries (no
         top-k) plus the sliding window. Global coarse context.

    Compute is FP32. Output shape is [N, d_model] (== input feature width).
    Attention is stateless across optimizer steps.
    """

    def __init__(
        self,
        d_model: int = 8,
        mode: str = 'csa',
        num_heads: int = 2,
        csa_compress: int = 4,
        csa_window: int = 8,
        csa_topk: int = 16,
        hca_compress: int = 128,
        indexer_rank: int = 4,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        assert mode in ('csa', 'hca'), f"mode must be 'csa' or 'hca', got {mode}"
        self.grad_checkpoint = grad_checkpoint
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.mode = mode
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.csa_compress = csa_compress
        self.csa_window = csa_window
        self.csa_topk = csa_topk
        self.hca_compress = hca_compress
        self.indexer_rank = indexer_rank

        # Q/K/V projections (multi-query: KV shared across heads, packed [d,d]).
        self.q_W = nn.Linear(d_model, d_model, bias=False)
        self.k_W = nn.Linear(d_model, d_model, bias=False)
        self.v_W = nn.Linear(d_model, d_model, bias=False)
        self.out_W = nn.Linear(d_model, d_model, bias=False)

        if mode == 'csa':
            # Learned pooling weights over the compression window.
            self.compress_w = nn.Parameter(torch.zeros(csa_window))
            # Lightning indexer: low-rank query + key projection.
            self.idx_DQ = nn.Parameter(torch.randn(d_model, indexer_rank) * 0.02)
            self.idx_UQ = nn.Parameter(torch.randn(indexer_rank, d_model) * 0.02)
            self.idx_K = nn.Parameter(torch.randn(d_model, indexer_rank) * 0.02)
        else:
            # HCA registers no compress_w / indexer params (mean pool, dense).
            self.register_parameter('compress_w', None)
            self.register_parameter('idx_DQ', None)
            self.register_parameter('idx_UQ', None)
            self.register_parameter('idx_K', None)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [L, d_model] -> [num_heads, L, head_dim]
        L = t.shape[0]
        return t.reshape(L, self.num_heads, self.head_dim).permute(1, 0, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, d_model] (|g|-sorted) → ctx: [N, d_model]. Stateless."""
        x = x.float()
        N, d = x.shape
        if N == 0:
            return torch.zeros(0, d, device=x.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(self.head_dim)

        # Per-token projections.
        q = self.q_W(x)                       # [N, d]
        k_tok = self.k_W(x)                   # [N, d]
        v_tok = self.v_W(x)                   # [N, d]

        # The heavy attention body (compression + indexer + scores + softmax)
        # is wrapped as a self-contained closure so it can be recomputed in the
        # backward pass instead of being held in memory. Output is the
        # pre-out_W context [N, d].
        def _csa_body(x, q, k_tok, v_tok):
            stride = self.csa_compress
            win = self.csa_window
            # ── Strided weighted pooling of K/V into compressed entries ──
            # Compressed entry j pools x[j*stride : j*stride+win].
            nc = (N + stride - 1) // stride   # ceil(N / stride)
            pool_w = torch.softmax(self.compress_w, dim=0)  # [win]
            starts = torch.arange(nc, device=x.device) * stride       # [Nc]
            offs = torch.arange(win, device=x.device)                 # [win]
            gather = starts.unsqueeze(1) + offs.unsqueeze(0)          # [Nc, win]
            valid = gather < N                                        # [Nc, win]
            gather_c = gather.clamp(max=N - 1)
            w_eff = pool_w.unsqueeze(0) * valid.float()               # [Nc, win]
            w_eff = w_eff / w_eff.sum(dim=1, keepdim=True).clamp(min=1e-12)
            # Pool compressed K/V.
            c_k = (k_tok[gather_c] * w_eff.unsqueeze(-1)).sum(dim=1)  # [Nc, d]
            c_v = (v_tok[gather_c] * w_eff.unsqueeze(-1)).sum(dim=1)  # [Nc, d]

            # ── Lightning indexer top-k selection ──
            qI = (x @ self.idx_DQ) @ self.idx_UQ                      # [N, d]
            kI_tok = x @ self.idx_K @ self.idx_UQ                     # [N, d]
            c_kI = (kI_tok[gather_c] * w_eff.unsqueeze(-1)).sum(dim=1)  # [Nc, d]
            idx_scores = (qI @ c_kI.T) / math.sqrt(d)                 # [N, Nc]
            topk = min(self.csa_topk, nc)
            _, sel = idx_scores.topk(topk, dim=-1)                    # [N, topk]

            # Gather selected compressed K/V per query → [N, topk, d].
            sel_k = c_k[sel]
            sel_v = c_v[sel]
            sel_kh = sel_k.reshape(N, topk, self.num_heads, self.head_dim)
            sel_vh = sel_v.reshape(N, topk, self.num_heads, self.head_dim)
            qh = q.reshape(N, self.num_heads, self.head_dim)          # [N, H, hd]
            # Scores over selected compressed entries: [N, H, topk]
            comp_scores = torch.einsum('nhd,nkhd->nhk', qh, sel_kh) * scale

            # ── Causal sliding window over raw tokens ──
            wsz = min(win, N)
            woffs = torch.arange(wsz, device=x.device)               # [wsz]
            qpos = torch.arange(N, device=x.device).unsqueeze(1)     # [N, 1]
            win_idx = qpos - woffs.unsqueeze(0)                      # [N, wsz]
            win_valid = win_idx >= 0                                  # [N, wsz]
            win_idx_c = win_idx.clamp(min=0)
            win_k = k_tok[win_idx_c].reshape(N, wsz, self.num_heads, self.head_dim)
            win_v = v_tok[win_idx_c].reshape(N, wsz, self.num_heads, self.head_dim)
            win_scores = torch.einsum('nhd,nwhd->nhw', qh, win_k) * scale
            win_scores = win_scores.masked_fill(
                ~win_valid.unsqueeze(1), float('-inf'))

            # ── Joint softmax over (selected compressed ∪ window) ──
            all_scores = torch.cat([comp_scores, win_scores], dim=-1)  # [N,H,topk+wsz]
            attn = torch.softmax(all_scores, dim=-1)
            attn_c = attn[:, :, :topk]                                # [N, H, topk]
            attn_w = attn[:, :, topk:]                                # [N, H, wsz]
            ctx_h = (torch.einsum('nhk,nkhd->nhd', attn_c, sel_vh)
                     + torch.einsum('nhw,nwhd->nhd', attn_w, win_v))  # [N, H, hd]
            ctx = ctx_h.reshape(N, d)
            return ctx

        def _hca_body(x, q, k_tok, v_tok):
            # ── HCA: stride-128 mean pool, dense attention over all entries ──
            stride = self.hca_compress
            nh = (N + stride - 1) // stride
            starts = torch.arange(nh, device=x.device) * stride       # [Nh]
            offs = torch.arange(stride, device=x.device)              # [stride]
            gather = starts.unsqueeze(1) + offs.unsqueeze(0)          # [Nh, stride]
            valid = gather < N
            gather_c = gather.clamp(max=N - 1)
            w_eff = valid.float()
            w_eff = w_eff / w_eff.sum(dim=1, keepdim=True).clamp(min=1e-12)
            c_k = (k_tok[gather_c] * w_eff.unsqueeze(-1)).sum(dim=1)  # [Nh, d]
            c_v = (v_tok[gather_c] * w_eff.unsqueeze(-1)).sum(dim=1)  # [Nh, d]

            qh = self._split_heads(q)                                # [H, N, hd]
            c_kh = self._split_heads(c_k)                            # [H, Nh, hd]
            c_vh = self._split_heads(c_v)                            # [H, Nh, hd]
            # Dense scores over all compressed entries: [H, N, Nh]
            comp_scores = torch.einsum('hnd,hmd->hnm', qh, c_kh) * scale

            # Causal sliding window (reuse csa_window size for local context).
            win = min(self.csa_window, N)
            woffs = torch.arange(win, device=x.device)
            qpos = torch.arange(N, device=x.device).unsqueeze(1)
            win_idx = qpos - woffs.unsqueeze(0)                      # [N, win]
            win_valid = win_idx >= 0
            win_idx_c = win_idx.clamp(min=0)
            win_kh = self._split_heads(k_tok)[:, win_idx_c, :]      # [H, N, win, hd]
            win_vh = self._split_heads(v_tok)[:, win_idx_c, :]
            win_scores = torch.einsum('hnd,hnwd->hnw', qh, win_kh) * scale
            win_scores = win_scores.masked_fill(
                ~win_valid.unsqueeze(0), float('-inf'))

            all_scores = torch.cat([comp_scores, win_scores], dim=-1)  # [H,N,Nh+win]
            attn = torch.softmax(all_scores, dim=-1)
            attn_c = attn[:, :, :nh]                                  # [H, N, Nh]
            attn_w = attn[:, :, nh:]                                  # [H, N, win]
            ctx_h = (torch.einsum('hnm,hmd->hnd', attn_c, c_vh)
                     + torch.einsum('hnw,hnwd->hnd', attn_w, win_vh))  # [H, N, hd]
            ctx = ctx_h.permute(1, 0, 2).reshape(N, d)
            return ctx

        body = _csa_body if self.mode == 'csa' else _hca_body
        ctx = _maybe_checkpoint_fn(
            body, self.grad_checkpoint, self.training, x, q, k_tok, v_tok)
        return self.out_W(ctx)


class MiniGRU(nn.Module):
    """Tiny per-element GRU for temporal memory across optimizer steps."""

    def __init__(self, input_dim: int, hidden_dim: int = 4,
                 grad_checkpoint: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grad_checkpoint = grad_checkpoint
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        def _gru_body(x, h):
            xh = torch.cat([x, h], dim=-1)
            z = torch.sigmoid(self.W_z(xh))
            r = torch.sigmoid(self.W_r(xh))
            xrh = torch.cat([x, r * h], dim=-1)
            h_tilde = torch.tanh(self.W_h(xrh))
            h_new = (1 - z) * h + z * h_tilde
            return h_new

        return _maybe_checkpoint_fn(
            _gru_body, self.grad_checkpoint, self.training, x, h)


class CSAHCAMetaNet(nn.Module):
    """CSA/HCA Hybrid Attention + 4-Head PEER + Per-Element GRU meta-network.

    The sequence mixer is a DeepSeek-V4-style pair of compressed attention
    blocks (CSA + HCA) replacing the previous bidirectional Mamba-3 scan. The
    GRU + PEER routing + expert MLP + Adam apply tail are unchanged.
    """

    def __init__(
        self,
        d_model: int = 8,
        d_state: int = 16,             # retained for back-compat config (unused by attn)
        mamba_expand: int = 2,         # retained for back-compat config (unused by attn)
        num_peer_heads: int = 4,
        peer_topk: int = 4,
        num_experts: int = 144,
        expert_hidden: int = 16,
        gru_hidden: int = 4,
        rescale: float = 0.1,
        recycle_interval: int = 100,
        recycle_threshold: float = 0.001,
        # CSA/HCA attention config
        n_heads: int = 2,
        csa_compress: int = 4,
        csa_window: int = 8,
        csa_topk: int = 16,
        hca_compress: int = 128,
        indexer_rank: int = 4,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.d_model = d_model
        self.d_state = d_state
        self.num_peer_heads = num_peer_heads
        # PEER routing top-k experts per head (spec §2). 4 is the design
        # default (4 product-key sub-selections per A/B axis → top-k soft
        # routing); 1 reproduces the old hard argmax (top-1) routing.
        self.peer_topk = max(1, int(peer_topk))
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.gru_hidden = gru_hidden
        self.rescale = rescale
        self.recycle_interval = recycle_interval
        self.recycle_threshold = recycle_threshold

        # CSA/HCA attention config
        self.n_heads = n_heads
        self.csa_compress = csa_compress
        self.csa_window = csa_window
        self.csa_topk = csa_topk
        self.hca_compress = hca_compress
        self.indexer_rank = indexer_rank

        self.pk_dim = int(math.sqrt(num_experts))
        assert self.pk_dim * self.pk_dim == num_experts, \
            f"num_experts must be perfect square, got {num_experts}"

        self.input_proj = nn.Linear(2, d_model, bias=True)
        # Sequence mixer: CSA (fine/local) + HCA (global/coarse) attention.
        self.csa_layer = HybridCompressedAttention(
            d_model, mode='csa', num_heads=n_heads,
            csa_compress=csa_compress, csa_window=csa_window, csa_topk=csa_topk,
            hca_compress=hca_compress, indexer_rank=indexer_rank,
            grad_checkpoint=grad_checkpoint,
        )
        self.hca_layer = HybridCompressedAttention(
            d_model, mode='hca', num_heads=n_heads,
            csa_compress=csa_compress, csa_window=csa_window, csa_topk=csa_topk,
            hca_compress=hca_compress, indexer_rank=indexer_rank,
            grad_checkpoint=grad_checkpoint,
        )

        gru_input_dim = 2 + 2 * d_model
        self.gru = MiniGRU(gru_input_dim, gru_hidden,
                           grad_checkpoint=grad_checkpoint)

        peer_input_dim = gru_hidden + 2 * d_model + 2
        self.peer_queries = nn.ModuleList([
            nn.Linear(peer_input_dim, d_model, bias=False)
            for _ in range(num_peer_heads)
        ])
        self.product_keys_A = nn.ParameterList([
            nn.Parameter(torch.randn(self.pk_dim, d_model // 2) * 0.02)
            for _ in range(num_peer_heads)
        ])
        self.product_keys_B = nn.ParameterList([
            nn.Parameter(torch.randn(self.pk_dim, d_model // 2) * 0.02)
            for _ in range(num_peer_heads)
        ])

        self.expert_W1 = nn.Parameter(torch.randn(num_experts, expert_hidden, 1) * 0.02)
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, expert_hidden))
        self.expert_W2 = nn.Parameter(torch.randn(num_experts, 1, expert_hidden) * 0.02)
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, 1))

        self.register_buffer('expert_counts', torch.zeros(num_experts, dtype=torch.int32))
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        grad: torch.Tensor,
        sharpness: torch.Tensor,
        gru_state: torch.Tensor,
        mamba_fwd_state: Optional[torch.Tensor] = None,
        mamba_bwd_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: mamba_fwd_state / mamba_bwd_state are accepted for signature
        # back-compat but ignored — CSA/HCA attention is stateless across steps.
        N = grad.numel()
        g = grad.reshape(-1).float()
        s = sharpness.reshape(-1).float()

        sort_idx = g.abs().argsort()
        g_sorted = g[sort_idx]
        s_sorted = s[sort_idx]

        inp = torch.stack([g_sorted, s_sorted], dim=-1)
        x = self.input_proj(inp)

        # CSA → fine/local context; HCA → global/coarse context (both on |g|-sorted x).
        csa_out = self.csa_layer(x)
        hca_out = self.hca_layer(x)

        unsort_idx = sort_idx.argsort()
        csa_ctx = csa_out[unsort_idx]
        hca_ctx = hca_out[unsort_idx]

        gru_input = torch.cat([
            g.unsqueeze(-1), s.unsqueeze(-1),
            csa_ctx, hca_ctx
        ], dim=-1)
        new_gru = self.gru(gru_input, gru_state.float())

        peer_input = torch.cat([
            new_gru, csa_ctx, hca_ctx,
            g.unsqueeze(-1), s.unsqueeze(-1)
        ], dim=-1)

        total_expert_out = torch.zeros(N, 1, device=grad.device, dtype=torch.float32)
        topk = self.peer_topk

        for h in range(self.num_peer_heads):
            query = self.peer_queries[h](peer_input)
            q_a = query[:, :self.d_model // 2]
            q_b = query[:, self.d_model // 2:]

            scores_a = q_a @ self.product_keys_A[h].T
            scores_b = q_b @ self.product_keys_B[h].T

            # Top-k product-key selection (spec §2). top-k=1 reduces to the old
            # hard argmax routing; the default top-k=4 keeps 4 sub-selections on
            # each axis and softmax-weights the k*k experts they index.
            ka = min(topk, scores_a.shape[-1])
            kb = min(topk, scores_b.shape[-1])
            top_a_vals, top_a_idx = scores_a.topk(ka, dim=-1)
            top_b_vals, top_b_idx = scores_b.topk(kb, dim=-1)

            soft_a = torch.softmax(top_a_vals * 10.0, dim=-1)
            soft_b = torch.softmax(top_b_vals * 10.0, dim=-1)

            expert_idx = (top_a_idx.unsqueeze(2) * self.pk_dim
                          + top_b_idx.unsqueeze(1)).reshape(N, -1)
            routing_weights = (soft_a.unsqueeze(2)
                               * soft_b.unsqueeze(1)).reshape(N, -1)
            num_active = ka * kb

            if self.training:
                with torch.no_grad():
                    self.expert_counts.scatter_add_(
                        0, expert_idx.reshape(-1),
                        torch.ones(expert_idx.numel(),
                                   dtype=torch.int32, device=expert_idx.device))

            W1 = self.expert_W1[expert_idx]
            b1 = self.expert_b1[expert_idx]
            W2 = self.expert_W2[expert_idx]
            b2 = self.expert_b2[expert_idx]

            g_exp = g.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(
                -1, num_active, -1, -1)
            z = torch.relu(torch.matmul(W1, g_exp).squeeze(-1) + b1)
            out = (torch.matmul(W2, z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                   + b2.squeeze(-1))
            head_out = (routing_weights * out).sum(dim=1, keepdim=True)
            total_expert_out = total_expert_out + head_out

        total_expert_out = total_expert_out / self.num_peer_heads
        smart_grad = (g.unsqueeze(-1) + self.rescale * total_expert_out).squeeze(-1)
        smart_grad = smart_grad.reshape(grad.shape).to(grad.dtype)

        # Attention is stateless: return None for the (legacy) scan states.
        return smart_grad, new_gru, None, None

    def forward_for_bilevel(
        self, grad, sharpness, gru_state,
        mamba_fwd_state=None, mamba_bwd_state=None,
    ):
        """Differentiable forward with top-k sparse soft PEER routing."""
        N = grad.numel()
        g = grad.reshape(-1).float()
        s = sharpness.reshape(-1).float()

        sort_idx = g.abs().argsort()
        g_sorted = g[sort_idx]
        s_sorted = s[sort_idx]

        inp = torch.stack([g_sorted, s_sorted], dim=-1)
        x = self.input_proj(inp)

        # CSA → fine/local context; HCA → global/coarse context (stateless).
        csa_out = self.csa_layer(x)
        hca_out = self.hca_layer(x)

        unsort_idx = sort_idx.argsort()
        csa_ctx = csa_out[unsort_idx]
        hca_ctx = hca_out[unsort_idx]

        gru_input = torch.cat([g.unsqueeze(-1), s.unsqueeze(-1), csa_ctx, hca_ctx], dim=-1)
        new_gru = self.gru(gru_input, gru_state.float())

        peer_input = torch.cat([new_gru, csa_ctx, hca_ctx, g.unsqueeze(-1), s.unsqueeze(-1)], dim=-1)

        total_expert_out = torch.zeros(N, 1, device=grad.device, dtype=torch.float32)
        topk = self.peer_topk

        for h in range(self.num_peer_heads):
            query = self.peer_queries[h](peer_input)
            q_a = query[:, :self.d_model // 2]
            q_b = query[:, self.d_model // 2:]

            scores_a = q_a @ self.product_keys_A[h].T
            scores_b = q_b @ self.product_keys_B[h].T

            ka = min(topk, scores_a.shape[-1])
            kb = min(topk, scores_b.shape[-1])
            top_a_vals, top_a_idx = scores_a.topk(ka, dim=-1)
            top_b_vals, top_b_idx = scores_b.topk(kb, dim=-1)

            soft_a = torch.softmax(top_a_vals * 10.0, dim=-1)
            soft_b = torch.softmax(top_b_vals * 10.0, dim=-1)

            expert_indices = (top_a_idx.unsqueeze(2) * self.pk_dim + top_b_idx.unsqueeze(1)).reshape(N, -1)
            routing_weights = (soft_a.unsqueeze(2) * soft_b.unsqueeze(1)).reshape(N, -1)

            W1 = self.expert_W1[expert_indices]
            b1 = self.expert_b1[expert_indices]
            W2 = self.expert_W2[expert_indices]
            b2 = self.expert_b2[expert_indices]

            num_active = ka * kb

            # Per-head expert MLP (the two batched matmuls + relu) is the
            # largest self-contained sub-block here; checkpoint it so the
            # [N, num_active, ...] activations are recomputed in backward.
            def _expert_mlp(g, W1, b1, W2, b2, routing_weights):
                g_exp = g.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, num_active, -1, -1)
                z = torch.relu(torch.matmul(W1, g_exp).squeeze(-1) + b1)
                out = torch.matmul(W2, z.unsqueeze(-1)).squeeze(-1).squeeze(-1) + b2.squeeze(-1)
                return (routing_weights * out).sum(dim=1, keepdim=True)

            head_out = _maybe_checkpoint_fn(
                _expert_mlp, self.grad_checkpoint, self.training,
                g, W1, b1, W2, b2, routing_weights)
            total_expert_out = total_expert_out + head_out

        total_expert_out = total_expert_out / self.num_peer_heads
        smart_grad = (g.unsqueeze(-1) + self.rescale * total_expert_out).squeeze(-1)
        # Attention is stateless: return None for the (legacy) scan states.
        return smart_grad.reshape(grad.shape).to(grad.dtype), new_gru, None, None

    @torch.no_grad()
    def _recycle_dead_experts(self):
        """Replace dead experts with mutated clones of top performers."""
        total_activations = self.expert_counts.sum().item()
        if total_activations == 0:
            return

        fractions = self.expert_counts.float() / total_activations
        dead_mask = fractions < self.recycle_threshold

        if not dead_mask.any():
            self.expert_counts.zero_()
            return

        counts_f = self.expert_counts.float()
        # Keep the sampled donor expert as a 0-d device tensor (no .item()
        # host sync); advanced indexing with it stays on device.
        top_expert = torch.multinomial(counts_f, 1)[0]
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        noise_scale = 0.01
        # Vectorized expert mutation: gather every dead row in one shot,
        # generate all the perturbation noise with a single randn_like per
        # tensor, and scatter back — replaces the per-index Python loop and
        # its per-iteration .item()/randn_like calls.
        for buf in (self.expert_W1, self.expert_b1,
                    self.expert_W2, self.expert_b2):
            donor = buf.data[top_expert].unsqueeze(0)  # [1, ...]
            slice_ = buf.data[dead_indices]
            buf.data[dead_indices] = donor + noise_scale * torch.randn_like(
                slice_)

        # Product-key recycling: a query/key row is reset only when *all* of
        # its experts are dead. Compute those conditions in a vectorized,
        # on-device fashion (per (a_idx,b_idx) group) instead of inside the
        # per-dead-expert Python loop. The boolean reductions below stay as
        # device tensors and are only realised by the indexed assignment.
        if self.num_experts % self.pk_dim == 0:
            num_a = self.num_experts // self.pk_dim
            # dead_mask reshaped to [num_a, pk_dim]: rows index a_idx, cols b_idx
            dm2d = dead_mask[:num_a * self.pk_dim].reshape(num_a, self.pk_dim)
            dead_rows = dm2d.all(dim=1).nonzero(as_tuple=True)[0]   # a_idx fully dead
            dead_cols = dm2d.all(dim=0).nonzero(as_tuple=True)[0]   # b_idx fully dead
            if dead_rows.numel() > 0:
                for h in range(self.num_peer_heads):
                    sel = self.product_keys_A[h].data[dead_rows]
                    self.product_keys_A[h].data[dead_rows] = \
                        torch.randn_like(sel) * 0.02
            if dead_cols.numel() > 0:
                for h in range(self.num_peer_heads):
                    sel = self.product_keys_B[h].data[dead_cols]
                    self.product_keys_B[h].data[dead_cols] = \
                        torch.randn_like(sel) * 0.02
        else:
            # Fallback (ragged num_experts): preserve the original per-dead
            # logic exactly, including its row/col .all() semantics.
            for idx in dead_indices:
                i = int(idx)
                a_idx = i // self.pk_dim
                b_idx = i % self.pk_dim
                row_start = a_idx * self.pk_dim
                row_end = row_start + self.pk_dim
                if dead_mask[row_start:row_end].all():
                    for h in range(self.num_peer_heads):
                        self.product_keys_A[h].data[a_idx] = torch.randn_like(
                            self.product_keys_A[h].data[a_idx]) * 0.02
                col_indices = torch.arange(0, self.num_experts, self.pk_dim,
                                           device=dead_mask.device) + b_idx
                col_indices = col_indices[col_indices < self.num_experts]
                if dead_mask[col_indices].all():
                    for h in range(self.num_peer_heads):
                        self.product_keys_B[h].data[b_idx] = torch.randn_like(
                            self.product_keys_B[h].data[b_idx]) * 0.02

        self.expert_counts.zero_()

    @property
    def has_cuda_bilevel(self):
        """Whether CUDA bilevel backward kernels are available."""
        return _HAS_CUDA_BACKWARD and next(self.parameters()).is_cuda

    def forward_for_bilevel_cuda(
        self, grad, sharpness, gru_state,
        mamba_fwd_state=None, mamba_bwd_state=None,
    ):
        """Bilevel forward with saved intermediates for the C++ backward.

        Runs the full meta-net forward under torch.no_grad(), saving the
        activation tensors the C++ bilevel_backward_driver needs. Returns
        (smart_grad, new_gru_state, saved_dict).
        """
        N = grad.numel()
        g = grad.reshape(-1).float()
        s = sharpness.reshape(-1).float()

        sort_idx = g.abs().argsort()
        g_sorted = g[sort_idx]
        s_sorted = s[sort_idx]

        inp = torch.stack([g_sorted, s_sorted], dim=-1)
        x = self.input_proj(inp)

        csa_out = self.csa_layer(x)
        hca_out = self.hca_layer(x)

        unsort_idx = sort_idx.argsort()
        csa_ctx = csa_out[unsort_idx]
        hca_ctx = hca_out[unsort_idx]

        gru_input = torch.cat([g.unsqueeze(-1), s.unsqueeze(-1),
                               csa_ctx, hca_ctx], dim=-1)
        h_old = gru_state.float()
        if h_old.dim() == 1:
            h_old = h_old.unsqueeze(0).expand(N, -1).contiguous()
        xh = torch.cat([gru_input, h_old], dim=-1)
        gru_z = torch.sigmoid(self.gru.W_z(xh))
        gru_r = torch.sigmoid(self.gru.W_r(xh))
        xrh = torch.cat([gru_input, gru_r * h_old], dim=-1)
        gru_h_tilde = torch.tanh(self.gru.W_h(xrh))
        new_gru = (1.0 - gru_z) * h_old + gru_z * gru_h_tilde

        peer_input = torch.cat([new_gru, csa_ctx, hca_ctx,
                                g.unsqueeze(-1), s.unsqueeze(-1)], dim=-1)

        total_expert_out = torch.zeros(N, 1, device=grad.device,
                                       dtype=torch.float32)
        topk = self.peer_topk
        for h in range(self.num_peer_heads):
            query = self.peer_queries[h](peer_input)
            q_a = query[:, :self.d_model // 2]
            q_b = query[:, self.d_model // 2:]
            scores_a = q_a @ self.product_keys_A[h].T
            scores_b = q_b @ self.product_keys_B[h].T
            ka = min(topk, scores_a.shape[-1])
            kb = min(topk, scores_b.shape[-1])
            top_a_vals, top_a_idx = scores_a.topk(ka, dim=-1)
            top_b_vals, top_b_idx = scores_b.topk(kb, dim=-1)
            soft_a = torch.softmax(top_a_vals * 10.0, dim=-1)
            soft_b = torch.softmax(top_b_vals * 10.0, dim=-1)
            expert_indices = (top_a_idx.unsqueeze(2) * self.pk_dim
                              + top_b_idx.unsqueeze(1)).reshape(N, -1)
            routing_weights = (soft_a.unsqueeze(2)
                               * soft_b.unsqueeze(1)).reshape(N, -1)
            W1 = self.expert_W1[expert_indices]
            b1 = self.expert_b1[expert_indices]
            W2 = self.expert_W2[expert_indices]
            b2 = self.expert_b2[expert_indices]
            num_active = ka * kb
            g_exp = g.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(
                -1, num_active, -1, -1)
            z_h = torch.relu(torch.matmul(W1, g_exp).squeeze(-1) + b1)
            out = (torch.matmul(W2, z_h.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                   + b2.squeeze(-1))
            head_out = (routing_weights * out).sum(dim=1, keepdim=True)
            total_expert_out = total_expert_out + head_out

        total_expert_out = total_expert_out / self.num_peer_heads
        smart_grad = (g.unsqueeze(-1)
                      + self.rescale * total_expert_out).squeeze(-1)
        smart_grad = smart_grad.reshape(grad.shape).to(grad.dtype)

        saved = {
            'sort_indices': sort_idx.detach(),
            'x_sorted': x.detach(),
            'csa_ctx': csa_ctx.detach(),
            'hca_ctx': hca_ctx.detach(),
            'gru_input': gru_input.detach(),
            'gru_h_old': h_old.detach(),
            'gru_z': gru_z.detach(),
            'gru_r': gru_r.detach(),
            'gru_h_tilde': gru_h_tilde.detach(),
            'peer_input': peer_input.detach(),
        }
        return smart_grad, new_gru.detach(), saved

    def get_weights(self, precision=None):
        """Extract all meta-model weights for the CUDA/HIP kernels.

        Returns the CSA/HCA weight set (spec §4): the Mamba scan weights are
        dropped and replaced by the CSA + HCA attention projections. The shared
        input_proj, GRU, PEER/product-key, and expert tensors are kept.

        Precision (spec §1):
          ``precision`` is an optional :class:`PrecisionConfig`. When ``None``
          (or ``projection_precision`` resolves to ``fp32``/``tf32``), every
          tensor is detached, FP32, contiguous — the historical behaviour that
          the FP32 kernel ABI consumes directly.

          When ``projection_precision == 'bf16'`` (the live SG2 default) the
          attention *projection* matrices (input_proj, CSA/HCA q/k/v/out,
          indexer DQ/UQ/K) are emitted in BF16 via
          :meth:`PrecisionConfig.convert_projection_weights`; ``'fp8'`` emits
          FP8 with a per-matrix ``<name>_scale`` companion. Compress logits,
          GRU, PEER queries, product keys, biases and the scan-free scalars stay
          FP32 (they are accumulators / tiny vectors, not GEMM operands). Expert
          MLP weights follow ``expert_precision`` (int8/int4 emit packed tensors
          + scales) — these are reachable, config-gated paths, not dead code.

        The emitted-dtype tensors are what the step() call site forwards to the
        kernel. See the ``_KERNEL_PROJECTION_DTYPE_ABI`` note in
        :meth:`SuperGrok2.step` for the documented C++ ABI dependency: the
        current ``supergrok2_*_step`` pybind signature is ``const float*`` for
        every projection weight, so the live fused tail upcasts the emitted
        low-precision tensors back to FP32 at that boundary until the bf16/fp8
        kernel overloads land. The bf16 *emission* path is fully reachable and
        is the dtype the eager / autograd meta-net consumes natively.
        """
        proj_mode = (precision.projection_precision
                     if precision is not None else 'fp32')
        expert_mode = (precision.expert_precision
                       if precision is not None else 'fp32')

        def _f32(t):
            return t.detach().float().contiguous()

        def _proj(t):
            """Emit a projection matrix at the configured precision.

            Returns (tensor, scale|None). FP32/TF32 → fp32 tensor, None.
            BF16 → bf16 tensor, None. FP8 → fp8 tensor + scalar scale.
            """
            if precision is None or proj_mode in ('fp32', 'tf32'):
                return _f32(t), None
            return precision.convert_projection_weights(t.detach())

        out = {}

        def _put_proj(name, t):
            w, scale = _proj(t)
            out[name] = w
            if scale is not None:
                out[name + '_scale'] = scale

        # ── shared input projection ──
        _put_proj('input_proj_W', self.input_proj.weight)
        out['input_proj_b'] = _f32(self.input_proj.bias)
        # ── CSA layer (produces csa_ctx) ──
        _put_proj('csa_q_W', self.csa_layer.q_W.weight)
        _put_proj('csa_k_W', self.csa_layer.k_W.weight)
        _put_proj('csa_v_W', self.csa_layer.v_W.weight)
        out['csa_compress_w'] = _f32(self.csa_layer.compress_w)
        _put_proj('csa_idx_DQ', self.csa_layer.idx_DQ)
        _put_proj('csa_idx_UQ', self.csa_layer.idx_UQ)
        _put_proj('csa_idx_K', self.csa_layer.idx_K)
        _put_proj('csa_out_W', self.csa_layer.out_W.weight)
        # ── HCA layer (produces hca_ctx) ──
        _put_proj('hca_q_W', self.hca_layer.q_W.weight)
        _put_proj('hca_k_W', self.hca_layer.k_W.weight)
        _put_proj('hca_v_W', self.hca_layer.v_W.weight)
        _put_proj('hca_out_W', self.hca_layer.out_W.weight)
        # ── GRU (carried across steps; FP32 accumulators) ──
        out['gru_W_z'] = _f32(self.gru.W_z.weight)
        out['gru_b_z'] = _f32(self.gru.W_z.bias)
        out['gru_W_r'] = _f32(self.gru.W_r.weight)
        out['gru_b_r'] = _f32(self.gru.W_r.bias)
        out['gru_W_h'] = _f32(self.gru.W_h.weight)
        out['gru_b_h'] = _f32(self.gru.W_h.bias)
        # ── PEER routing (queries/keys stay FP32) ──
        out['peer_queries'] = [_f32(q.weight) for q in self.peer_queries]
        out['product_keys_A'] = [_f32(k) for k in self.product_keys_A]
        out['product_keys_B'] = [_f32(k) for k in self.product_keys_B]
        # ── Expert MLP — follows expert_precision (fp32/int8/int4) ──
        if precision is None or expert_mode == 'fp32':
            out['expert_W1'] = _f32(self.expert_W1)
            out['expert_b1'] = _f32(self.expert_b1)
            out['expert_W2'] = _f32(self.expert_W2)
            out['expert_b2'] = _f32(self.expert_b2)
        else:
            # int8 / int4 packed expert weights + scales (reachable, gated).
            out['expert_quant'] = precision.convert_expert_weights(
                self.expert_W1.detach(), self.expert_b1.detach(),
                self.expert_W2.detach(), self.expert_b2.detach())
            # Keep FP32 copies too so the FP32 kernel ABI still has operands
            # while the int8/int4 expert kernel overload is not the live tail.
            out['expert_W1'] = _f32(self.expert_W1)
            out['expert_b1'] = _f32(self.expert_b1)
            out['expert_W2'] = _f32(self.expert_W2)
            out['expert_b2'] = _f32(self.expert_b2)
        # ── scalars / config ints ──
        out.update({
            'projection_precision': proj_mode,
            'expert_precision': expert_mode,
            'rescale': self.rescale,
            'd_model': self.d_model,
            'pk_dim': self.pk_dim,
            'expert_hidden': self.expert_hidden,
            'gru_hidden': self.gru_hidden,
            'num_peer_heads': self.num_peer_heads,
            'num_experts': self.num_experts,
            'num_heads': self.n_heads,
            'csa_compress': self.csa_compress,
            'csa_window': self.csa_window,
            'csa_topk': self.csa_topk,
            'hca_compress': self.hca_compress,
            'indexer_rank': self.indexer_rank,
            'topk': self.peer_topk,
        })
        return out


# Back-compat alias: the meta-net used to be named ``Mamba3PEERMetaNet``.
Mamba3PEERMetaNet = CSAHCAMetaNet


# ════════════════════════════════════════════════════════════════════════════
#  SuperGrok v2 optimizer
# ════════════════════════════════════════════════════════════════════════════


class SuperGrok2(Optimizer):
    r"""SuperGrok v2 — CSA/HCA Hybrid Attention + PEER Grokking Optimizer.

    Same dynamics as SuperGrok v1.5 (sigmoid gating, adaptive SAM/bilevel,
    progressive WD) but with a CSA/HCA hybrid attention + PEER meta-net that
    captures cross-element gradient correlations via DeepSeek-V4-style
    compressed sparse (local) + heavily-compressed dense (global) attention.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1.0,
        alpha_init: float = 0.98,
        lamb: float = 2.0,
        gamma: float = 0.1,
        gamma_alpha: float = 0.0,
        kappa: float = 0.1,
        warmup_steps: int = 100,
        warmup_ramp: int = 100,
        gradient_clipping: float = 1.0,
        meta_net: Optional[nn.Module] = None,
        d_model: int = 8,
        d_state: int = 16,
        mamba_expand: int = 2,
        num_peer_heads: int = 4,
        peer_topk: int = 4,
        num_experts: int = 144,
        expert_hidden: int = 16,
        gru_hidden: int = 4,
        meta_rescale: float = 0.1,
        # CSA/HCA attention config
        n_heads: int = 2,
        csa_compress: int = 4,
        csa_window: int = 8,
        csa_topk: int = 16,
        hca_compress: int = 128,
        indexer_rank: int = 4,
        recycle_interval: int = 100,
        recycle_threshold: float = 0.001,
        alpha_update_freq: int = 100,
        zero_loss_threshold: float = 1e-4,
        zero_acc_threshold: float = 0.995,
        sam_rho: float = 0.05,
        gate_scale: float = 20.0,
        gate_thresh: float = 0.8,
        sam_freq_min: int = 3,
        sam_freq_max: int = 20,
        sam_scale: float = 20.0,
        sam_thresh: float = 0.85,
        bilevel_freq_min: int = 5,
        bilevel_freq_max: int = 30,
        bilevel_scale: float = 20.0,
        bilevel_thresh: float = 0.9,
        wd_ramp: float = 4.0,
        wd_scale: float = 20.0,
        wd_thresh: float = 0.9,
        sam_enable_threshold: float = 0.0,
        bilevel_checkpoint_interval: int = 1,
        # Precision (spec §1): bf16 is the live default for the attention
        # projection GEMMs, configurable. fp8/int4 are reachable, opt-in.
        projection_precision: str = 'bf16',
        expert_precision: str = 'fp32',
        # Model config (spec §7): vocabulary / class count the optimizer's
        # model operates over. Exposed here so the kernel-arg plumbing can pass
        # it down instead of relying on a hardcoded 97/99 in the C++ side.
        vocab_size: int = 97,
        # Distributed training parameters
        bilevel_allreduce_meta_grads: bool = True,
        expert_allreduce_before_recycle: bool = True,
        mamba_state_sync_interval: int = 1000,
        state_precision: str = 'fp32',
        use_grad_hooks: bool = False,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            # LOUD unsupported (not a silent wrong): every SG2 step path
            # (step / step_compiled / batched) reads lr/betas/eps/wd from
            # param_groups[0] and would silently apply them to ALL groups.
            # SuperGrok11/15 honor per-group scalars via one fused call per
            # group; porting that across SG2's three dispatch paths is queued
            # work — until then, refuse rather than mis-apply.
            raise NotImplementedError(
                "SuperGrok2 currently supports a single param group "
                f"(got {len(self.param_groups)}): the fused step paths read "
                "hyperparameters from param_groups[0] only. Merge your groups "
                "or use per-parameter masks; per-group support is tracked.")
        self.state_precision = state_precision

        # NOTE (owner directive — no functionality suppression): a
        # `meta_gate_power` memorization-ratchet knob was REMOVED here (it
        # scaled base_alpha toward 0 post-memorization = suppression, and was
        # moot for SG2 anyway: on-device isolation shows SG2 destabilizes
        # *before* memorization via the grokfast `lamb` amplification schedule
        # — lamb_eff spikes at train_acc≈gate_thresh — while the base Adam core
        # is stable). The fix path is TUNING the schedule scalars (lamb,
        # gate_scale, gate_thresh, meta_rescale, sg2_meta_lr) with everything
        # active — see tuning/tune_optimizers.py.

        self.alpha_init = alpha_init
        self.sam_enable_threshold = sam_enable_threshold
        self.lamb = lamb
        self.gamma = gamma
        self.gamma_alpha = gamma_alpha
        self.kappa = kappa
        self.warmup_steps = warmup_steps
        self.warmup_ramp = max(1, warmup_ramp)
        self.gradient_clipping = gradient_clipping
        self.alpha_update_freq = alpha_update_freq
        self.zero_loss_threshold = zero_loss_threshold
        self.zero_acc_threshold = zero_acc_threshold
        self.sam_rho = sam_rho

        # Precision configuration for projection GEMMs (spec §1). bf16 is the
        # live default; fp8/int4 are reachable, opt-in. Threaded into
        # get_weights() so the chosen-precision tensors actually reach the
        # kernel call boundary (see step()).
        self.precision_config = PrecisionConfig(
            projection_precision=projection_precision,
            expert_precision=expert_precision)

        # Meta-net hyperparams
        self.d_model = d_model
        self.d_state = d_state
        self.num_peer_heads = num_peer_heads
        self.peer_topk = max(1, int(peer_topk))
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.gru_hidden = gru_hidden
        self.meta_rescale = meta_rescale
        # Model vocab/class count (spec §7) — passed down to the kernel args.
        self.vocab_size = int(vocab_size)

        # CSA/HCA attention config
        self.n_heads = n_heads
        self.csa_compress = csa_compress
        self.csa_window = csa_window
        self.csa_topk = csa_topk
        self.hca_compress = hca_compress
        self.indexer_rank = indexer_rank

        # Adaptive scheduling params
        self.gate_scale = gate_scale
        self.gate_thresh = gate_thresh
        self.sam_freq_min = sam_freq_min
        self.sam_freq_max = sam_freq_max
        self.sam_scale = sam_scale
        self.sam_thresh = sam_thresh
        self.bilevel_freq_min = bilevel_freq_min
        self.bilevel_freq_max = bilevel_freq_max
        self.bilevel_scale = bilevel_scale
        self.bilevel_thresh = bilevel_thresh
        self.wd_ramp = wd_ramp
        self.wd_scale = wd_scale
        self.wd_thresh = wd_thresh
        self.bilevel_checkpoint_interval = max(1, bilevel_checkpoint_interval)

        # Distributed training configuration
        self.bilevel_allreduce_meta_grads = bilevel_allreduce_meta_grads
        self.expert_allreduce_before_recycle = expert_allreduce_before_recycle
        self.mamba_state_sync_interval = mamba_state_sync_interval

        # Meta-net: CSA/HCA Hybrid Attention + 4-Head PEER + GRU
        if meta_net is None:
            self.meta_net = CSAHCAMetaNet(
                d_model=d_model,
                d_state=d_state,
                mamba_expand=mamba_expand,
                num_peer_heads=num_peer_heads,
                peer_topk=self.peer_topk,
                num_experts=num_experts,
                expert_hidden=expert_hidden,
                gru_hidden=gru_hidden,
                rescale=meta_rescale,
                recycle_interval=recycle_interval,
                recycle_threshold=recycle_threshold,
                n_heads=n_heads,
                csa_compress=csa_compress,
                csa_window=csa_window,
                csa_topk=csa_topk,
                hca_compress=hca_compress,
                indexer_rank=indexer_rank,
            )
        else:
            self.meta_net = meta_net

        try:
            first_param = next(iter(self.param_groups[0]["params"]))
            self.meta_net = self.meta_net.to(first_param.device)
        except (StopIteration, IndexError):
            first_param = None

        # JIT specialization removed in the all-specialized refactor.
        # Per-arch kernels in csrc/kernels/{cuda,hip}/<arch>/ replace the
        # runtime JIT path. Keep these attributes None for any code that
        # checks for them.
        self._jit = None
        self._jit_kernels = None

        self._global_step = 0
        self._step_counter = 0  # Python int for expert recycling (avoids GPU sync)
        self._cached_alpha = alpha_init
        self._cached_train_acc = 0.0

        # Build flat parameter lists
        self._flat_params = []
        self._flat_steps = []
        self._flat_layer_alphas = []
        self._flat_layer_beta1s = []
        self._flat_exp_avgs = []
        self._flat_exp_avg_sqs = []
        self._flat_exp_avg_scales = []  # Config 3: INT8 per-block scales
        self._flat_mus = []
        self._flat_slows = []  # slow-gradient EMA (restored grokfast term)
        self._flat_sharpness = []
        self._flat_gru_states = []
        self._flat_mamba_fwd_states = []
        self._flat_mamba_bwd_states = []
        self._param_to_idx = {}

        idx = 0
        num_params = sum(1 for g in self.param_groups for _ in g["params"])
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                self._flat_params.append(p)
                self._flat_steps.append(0)
                lb1 = beta1 * ((1.0 - gamma) ** idx)
                self._flat_layer_beta1s.append(lb1)
                if gamma_alpha == 0.0:
                    la_factor = 1.0
                else:
                    max_idx = max(num_params - 1, 1)
                    la_factor = (1.0 - gamma_alpha) ** (max_idx - idx)
                self._flat_layer_alphas.append(la_factor)
                self._param_to_idx[id(p)] = idx
                idx += 1

        self._num_params = num_params
        self._state_initialized = False
        self._flat_param_data = [p.data for p in self._flat_params]
        self._weights_dirty = True
        self._cached_weights = None

        self._use_grad_hooks = use_grad_hooks
        if use_grad_hooks:
            _register_grad_hooks(self)

    def _ensure_state(self):
        if self._state_initialized:
            return
        for p in self._flat_params:
            N = p.data.numel()
            if self.state_precision == 'config3':
                # Config 3: INT8 per-block for exp_avg, BF16 for others
                block_size = 32
                num_blocks = (N + block_size - 1) // block_size
                self._flat_exp_avgs.append(
                    torch.zeros(N, dtype=torch.int8, device=p.device))
                self._flat_exp_avg_scales.append(
                    torch.zeros(num_blocks, dtype=torch.float32, device=p.device))
                self._flat_exp_avg_sqs.append(
                    torch.zeros(N, dtype=torch.bfloat16, device=p.device))
                self._flat_mus.append(
                    torch.zeros(N, dtype=torch.bfloat16, device=p.device))
                # Slow-gradient EMA buffer (restored grokfast term). Allocated
                # exactly like _flat_mus so the kernel's sg2_apply_step has a
                # dedicated `slow` accumulator distinct from the expert-EMA mu.
                self._flat_slows.append(
                    torch.zeros(N, dtype=torch.bfloat16, device=p.device))
                self._flat_sharpness.append(
                    torch.zeros(N, dtype=torch.bfloat16, device=p.device))
                self._flat_gru_states.append(
                    torch.zeros(N, self.gru_hidden,
                                dtype=torch.bfloat16, device=p.device))
            else:
                # FP32 (default)
                self._flat_exp_avgs.append(
                    torch.zeros(N, dtype=torch.float32, device=p.device))
                self._flat_exp_avg_sqs.append(
                    torch.zeros(N, dtype=torch.float32, device=p.device))
                self._flat_mus.append(
                    torch.zeros(N, dtype=torch.float32, device=p.device))
                # Slow-gradient EMA buffer (restored grokfast term). Allocated
                # exactly like _flat_mus so the kernel's sg2_apply_step has a
                # dedicated `slow` accumulator distinct from the expert-EMA mu.
                self._flat_slows.append(
                    torch.zeros(N, dtype=torch.float32, device=p.device))
                self._flat_sharpness.append(
                    torch.zeros(N, dtype=torch.float32, device=p.device))
                self._flat_gru_states.append(
                    torch.zeros(N, self.gru_hidden,
                                dtype=torch.float32, device=p.device))
            # Back-compat placeholders: CSA/HCA attention is stateless, so these
            # per-parameter scan-state slots stay None (kept for any external
            # code that indexes the lists).
            self._flat_mamba_fwd_states.append(None)
            self._flat_mamba_bwd_states.append(None)
        self._state_initialized = True

    def _sigmoid(self, scale, value, thresh):
        return 1.0 / (1.0 + math.exp(-scale * (value - thresh)))

    def _update_alpha(self, train_loss, val_loss, train_acc):
        if train_loss is None and train_acc is None:
            return
        signal = 0.0
        memorized = False
        if train_acc is not None and train_acc >= self.zero_acc_threshold:
            memorized = True
        elif train_loss is not None and train_loss < self.zero_loss_threshold:
            memorized = True
        if memorized:
            signal = 10.0
        elif val_loss is not None and train_loss is not None and train_loss > 1e-12:
            signal = max(0.0, (val_loss - train_loss) / train_loss)
        self._cached_alpha = self.alpha_init * math.exp(-self.kappa * signal)

    def _get_ramp_factor(self):
        step = self._global_step
        if step <= self.warmup_steps:
            return 0.0
        elapsed = step - self.warmup_steps
        return min(1.0, elapsed / self.warmup_ramp)

    def _get_effective_wd(self, base_wd):
        acc = self._cached_train_acc
        sigmoid_val = self._sigmoid(self.wd_scale, acc, self.wd_thresh)
        return base_wd * (1.0 + self.wd_ramp * sigmoid_val)

    def _get_gate_signal(self):
        return self._sigmoid(self.gate_scale, self._cached_train_acc, self.gate_thresh)

    def _get_effective_sam_freq(self):
        if self._cached_train_acc < self.sam_enable_threshold:
            return 999999  # effectively disabled
        acc = self._cached_train_acc
        sam_heat = self._sigmoid(self.sam_scale, acc, self.sam_thresh)
        freq = self.sam_freq_max - (self.sam_freq_max - self.sam_freq_min) * sam_heat
        return max(1, round(freq))

    def _get_effective_bilevel_freq(self):
        acc = self._cached_train_acc
        bilevel_heat = self._sigmoid(self.bilevel_scale, acc, self.bilevel_thresh)
        freq = self.bilevel_freq_max - (self.bilevel_freq_max - self.bilevel_freq_min) * bilevel_heat
        return max(1, round(freq))

    def _is_distributed(self) -> bool:
        """Check if distributed training is active."""
        return dist.is_available() and dist.is_initialized()

    def _allreduce_meta_grads(self):
        """All-reduce meta-net gradients across ranks for consistent updates."""
        if not self._is_distributed():
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        for p in self.meta_net.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

    def _allreduce_expert_counts(self):
        """All-reduce expert activation counts across ranks before recycling."""
        if not self._is_distributed():
            return
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        dist.all_reduce(self.meta_net.expert_counts, op=dist.ReduceOp.SUM)

    def _sync_mamba_states(self):
        """No-op (back-compat). CSA/HCA attention is stateless across optimizer
        steps, so there is no carried scan state to broadcast across ranks. The
        GRU state is per-parameter and not synced (matching prior behavior)."""
        return

    @staticmethod
    def _is_fsdp_wrapped(model) -> bool:
        """Check if a model is wrapped with FSDP."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)

    @staticmethod
    def exclude_meta_net_from_fsdp(meta_net: nn.Module):
        """Mark meta-net parameters to be excluded from FSDP sharding.

        Call this before wrapping the model with FSDP. Sets
        ``_fsdp_wrap = False`` on the meta-net module so FSDP will
        not shard its parameters (they are small and must stay replicated).

        Usage::

            optimizer = SuperGrok2(model.parameters(), ...)
            SuperGrok2.exclude_meta_net_from_fsdp(optimizer.meta_net)
            model = FSDP(model, auto_wrap_policy=...)
        """
        meta_net._fsdp_wrap = False
        for module in meta_net.modules():
            module._fsdp_wrap = False

    def _gather_full_grad_fsdp(self, model):
        """Gather full gradients from FSDP-sharded parameters.

        Returns a list of (param_idx, full_grad) pairs for all active
        parameters. The full gradients are temporary tensors that will
        be used for the meta-net scan and then discarded.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        gathered = []
        with FSDP.summon_full_params(model, writeback=False):
            for i, p in enumerate(self._flat_params):
                if p.grad is not None:
                    gathered.append((i, p.grad.detach().clone()))
        return gathered

    # Projection-weight keys emitted by CSAHCAMetaNet.get_weights() that are
    # subject to the precision config (the GEMM operands). Compress logits /
    # GRU / PEER / experts are not in this set (they stay FP32 by design).
    _PROJECTION_WEIGHT_KEYS = (
        'input_proj_W',
        'csa_q_W', 'csa_k_W', 'csa_v_W',
        'csa_idx_DQ', 'csa_idx_UQ', 'csa_idx_K', 'csa_out_W',
        'hca_q_W', 'hca_k_W', 'hca_v_W', 'hca_out_W',
    )

    def _kernel_abi_view(self, emitted):
        """Return the weight bundle the kernel ABI consumes.

        The C++ kernels now accept both FP32 and BF16 projection weights
        (templated on weight type). BF16 tensors pass through unchanged,
        saving the bandwidth of the FP32 upcast. FP8 still requires the
        boundary upcast (no FP8 kernel overload yet).
        """
        proj_mode = emitted.get('projection_precision', 'fp32')
        if proj_mode in ('fp32', 'tf32', 'bf16'):
            return emitted  # FP32 and BF16 pass through natively
        # FP8 still needs the upcast until fp8 kernel overloads land
        w = dict(emitted)
        for key in self._PROJECTION_WEIGHT_KEYS:
            t = w.get(key)
            if t is not None and t.dtype != torch.float32:
                w[key] = t.float().contiguous()
        return w

    @torch.no_grad()
    def step(self, closure=None, train_loss=None, val_loss=None, train_acc=None):
        """Eager optimiser step — REMOVED (pure L3-TC).

        On the L3-TC path SuperGrok2's DEDICATED megakernel runs the REAL fwd+bwd
        + the FULL CSA/HCA/PEER/GRU meta-net AS the optimizer phase (in-kernel
        segmented sort + SAM 2nd backward + sg2_meta_stages) in ONE persistent
        launch; the meta-net is trained host-side via ``sam_step``/``bilevel_step``
        (pure-PyTorch autograd, kept) and the dedicated entry scatters its weights.
        The eager kernel dispatch here (the use_cuda batched path) is gone.
        """
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def sam_step(self, model, train_x, train_y, criterion):
        """SAM ascent step → per-parameter sharpness |g(w+e) − g(w)|.

        Recomputes the gradient at the SAM-perturbed point e(w)=rho·g/‖g‖ via
        ``torch.func.functional_call`` and writes the elementwise sharpness into
        ``_flat_sharpness`` (the buffer the meta-net's 2nd input and the fused
        kernel read). Mirrors the SuperGrok11 fix — the previous version was
        broken on **every** call (silently: the race loop swallows the exception),
        so SG2 ran with NO SAM signal, and because it ``model.zero_grad()``'d and
        then raised, it also left the params grad-less → starved ``bilevel_step``
        (the only meta-net trainer) of gradients. Two correctness points:
          • the perturbed params come from ``.detach()`` so they must be
            ``.requires_grad_(True)`` or autograd has nothing to differentiate
            (the old ``sam_loss.backward()`` raised "element 0 … does not require
            grad" every time);
          • ``functional_call`` substitutes the perturbed tensors as the module's
            params, so gradients live on THEM — take them with ``autograd.grad``
            and leave each ``p.grad`` (the original backward result ``step()``
            and ``bilevel_step`` consume) completely untouched (no zero_grad).
        Sharpness is flattened to ``(numel,)`` to match its ``torch.zeros(N)`` init.
        """
        self._ensure_state()
        train_grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                train_grads[name] = p.grad.detach().clone()
        if not train_grads:
            return 0.0

        flat_grads = [train_grads[n] for n, _ in model.named_parameters() if n in train_grads]
        total_norm_sq = sum(g.norm().pow(2) for g in flat_grads if g.numel() > 0)
        grad_norm = total_norm_sq.sqrt() + 1e-12
        rho_over_norm = self.sam_rho / grad_norm

        named_params = dict(model.named_parameters())
        pert_names = [n for n in named_params if n in train_grads]
        perturbed = {}
        for name, p in named_params.items():
            if name in train_grads:
                perturbed[name] = (p.detach() + rho_over_norm * train_grads[name]).requires_grad_(True)
            else:
                perturbed[name] = p.detach()

        with torch.enable_grad():
            sam_logits = torch.func.functional_call(model, perturbed, (train_x,))
            sam_loss = criterion(sam_logits, train_y)
            sam_grads = torch.autograd.grad(
                sam_loss, [perturbed[n] for n in pert_names], allow_unused=True)
        sam_loss_val = sam_loss.item()

        for name, sg in zip(pert_names, sam_grads):
            if sg is None:
                continue
            pidx = self._param_to_idx.get(id(named_params[name]))
            if pidx is not None:
                self._flat_sharpness[pidx] = (sg.detach() - train_grads[name]).abs().reshape(-1)

        return sam_loss_val

    def bilevel_step(self, model, train_x, train_y, val_x, val_y, criterion, meta_optimizer):
        """Bilevel meta-net training (pure-PyTorch autograd path).

        PURE L3-TC (owner directive): the C++ hand-written VJP backward
        (``_bilevel_step_cuda`` → ``_ops.supergrok2_bilevel_backward``) was DROPPED;
        this host-side meta-net training now ALWAYS uses the ``torch.autograd``
        path (``_bilevel_step_autograd``). The meta-net it trains is then evaluated
        IN the megakernel (fused_train_step scatters its weights). The model
        forward/backward + optimizer update never run eager here.
        """
        self._ensure_state()
        named_params = list(model.named_parameters())

        saved_grads = {}
        for name, p in named_params:
            if p.grad is not None:
                saved_grads[name] = p.grad.detach().clone()

        if not any(p.is_cuda for _, p in named_params if p.grad is not None):
            raise RuntimeError(
                "SuperGrok2.bilevel_step() requires CUDA tensors.")

        mn = self.meta_net

        param_info = []
        for name, p in named_params:
            if name not in saved_grads:
                continue
            pidx = self._param_to_idx.get(id(p))
            if pidx is None:
                continue
            grad_flat = saved_grads[name].reshape(-1).float().contiguous()
            sharp_flat = self._flat_sharpness[pidx].reshape(-1).float().contiguous()
            if grad_flat.numel() == 0:
                continue
            param_info.append((name, p, pidx, grad_flat, sharp_flat))

        if not param_info:
            # Nothing to do: no saved grad maps into the meta-net's param index
            # (or every mapped grad is 0-element). Return a 0-d zero loss so the
            # caller can accumulate it uniformly with the real return paths
            # (which return ``val_loss.detach()``). Place it on the device of any
            # saved grad when one exists, else CPU. (The previous
            # ``saved_grads and ... or 'cpu'`` and/or idiom was precedence-
            # fragile and keyed the device off saved_grads rather than this
            # branch's actual trigger; spelled out as if/else for clarity.)
            if saved_grads:
                dev = next(iter(saved_grads.values())).device
            else:
                dev = torch.device('cpu')
            return torch.zeros((), device=dev)

        # PURE L3-TC: the C++ bilevel backward (_bilevel_step_cuda) is DROPPED;
        # always use the autograd path. (mn.has_cuda_bilevel is no longer consulted.)
        return self._bilevel_step_autograd(
            model, val_x, val_y, criterion, meta_optimizer,
            named_params, saved_grads, param_info, mn)

    def _bilevel_step_autograd(self, model, val_x, val_y, criterion,
                               meta_optimizer, named_params, saved_grads,
                               param_info, mn):
        """Autograd fallback path for bilevel_step."""
        meta_optimizer.zero_grad()
        smart_grads = {}
        new_gru_states = {}
        with torch.enable_grad():
            for (name, p, pidx, grad_flat, sharp_flat) in param_info:
                gru_state = self._flat_gru_states[pidx].float()
                smart_grad, new_gru, _f, _b = mn.forward_for_bilevel(
                    grad_flat, sharp_flat, gru_state)
                smart_grads[name] = smart_grad
                new_gru_states[pidx] = new_gru.detach()

        model.zero_grad()
        with torch.enable_grad():
            val_loss = criterion(model(val_x), val_y)
            val_loss.backward()

        grad_outputs = []
        outputs = []
        for (name, p, pidx, grad_flat, sharp_flat) in param_info:
            if p.grad is None:
                continue
            # Sanitize the val-grad on-device instead of an
            # ``isfinite(vg).all()`` reduction-to-host skip: zero any NaN/Inf
            # entries (no-op when already finite). nan_to_num maps a fully
            # non-finite grad to all-zeros, whose norm is 0 -> the existing
            # ``vg_norm > 1e-12`` branch then leaves vg_unit at zero, i.e. no
            # contribution, matching the old skip semantics in that case.
            vg = torch.nan_to_num(
                p.grad.detach().reshape(-1).float(),
                nan=0.0, posinf=0.0, neginf=0.0)
            vg_norm = vg.norm()
            vg_unit = vg / vg_norm if vg_norm > 1e-12 else vg
            sg = smart_grads[name].reshape(-1)
            outputs.append(sg)
            grad_outputs.append(-vg_unit)

        if outputs:
            torch.autograd.backward(outputs, grad_outputs)

        if self.bilevel_allreduce_meta_grads:
            self._allreduce_meta_grads()

        meta_optimizer.step()
        self._weights_dirty = True

        for pidx, ng in new_gru_states.items():
            self._flat_gru_states[pidx] = ng

        for name, p in named_params:
            p.grad = saved_grads.get(name)

        return val_loss.detach()

    def _bilevel_step_cuda(self, model, val_x, val_y, criterion,
                           meta_optimizer, named_params, saved_grads,
                           param_info, mn):
        """C++ hand-written bilevel backward — DROPPED (pure L3-TC).

        The eager VJP kernel (``_ops.supergrok2_bilevel_backward``) is removed;
        ``bilevel_step`` now always routes to ``_bilevel_step_autograd``. This
        method is retained only so an explicit call fails loudly rather than
        silently running a removed eager path.
        """
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def bilevel_step_distributed(self, model, train_x, train_y, val_x, val_y,
                                  criterion, meta_optimizer, process_group=None):
        """Distributed-aware bilevel step with coordinated validation forward pass.

        All ranks must call this simultaneously. Each rank computes validation
        loss on its local shard, then validation gradients are all-reduced before
        computing meta-gradients. Meta-net gradients are also all-reduced before
        stepping the meta-optimizer.

        Args:
            model: The model (may be DDP or FSDP wrapped).
            train_x, train_y: Training batch (local shard).
            val_x, val_y: Validation batch (local shard).
            criterion: Loss function.
            meta_optimizer: Optimizer for meta-net parameters.
            process_group: Optional process group for communication.
        """
        if not self._is_distributed():
            return self.bilevel_step(
                model, train_x, train_y, val_x, val_y, criterion, meta_optimizer)

        # The regular bilevel_step already has all-reduce hooks for meta-grads
        # via self.bilevel_allreduce_meta_grads. We just need to ensure
        # the validation forward+backward is coordinated.
        return self.bilevel_step(
            model, train_x, train_y, val_x, val_y, criterion, meta_optimizer)

    def sam_meta_step(self, model, train_x, train_y, val_x, val_y, criterion, meta_optimizer):
        """Combined SAM + bilevel (backward-compatible)."""
        sam_loss = self.sam_step(model, train_x, train_y, criterion)
        val_loss = self.bilevel_step(model, train_x, train_y, val_x, val_y, criterion, meta_optimizer)
        return sam_loss, val_loss

    def _prepare_for_compile(self):
        """Pre-build static tensor lists for torch.compile / CUDA graph capture.

        Must be called after at least one eager step (so states are initialized).
        Freezes the parameter list and pre-extracts meta-net weights.
        After calling this, only ``step_compiled()`` should be used.
        """
        self._ensure_state()

        # Force weight extraction and cache (precision-aware emission +
        # FP32 kernel-ABI view, mirroring step()).
        def _f32c(t):
            t = t if t.dtype == torch.float32 else t.float()
            return t if t.is_contiguous() else t.contiguous()
        self._emitted_weights = self.meta_net.get_weights(self.precision_config)
        w = self._kernel_abi_view(self._emitted_weights)
        self._cached_peer_query_Ws = torch.stack(
            [_f32c(q.weight.data) for q in self.meta_net.peer_queries])
        self._cached_prod_keys_A = torch.stack(
            [_f32c(k.data) for k in self.meta_net.product_keys_A])
        self._cached_prod_keys_B = torch.stack(
            [_f32c(k.data) for k in self.meta_net.product_keys_B])
        self._cached_weights = w
        self._weights_dirty = False

        # Pre-build static lists of all parameters (assume all active)
        self._static_params = [p.data for p in self._flat_params]
        self._static_exp_avgs = list(self._flat_exp_avgs)
        self._static_exp_avg_sqs = list(self._flat_exp_avg_sqs)
        self._static_mus = list(self._flat_mus)
        self._static_slows = list(self._flat_slows)  # mirror _static_mus
        self._static_gru_states = list(self._flat_gru_states)
        # CSA/HCA attention is stateless — no static scan-state lists needed.

        # Pre-allocate static gradient buffers
        self._static_grads = [
            torch.zeros_like(p.data, dtype=torch.float32) for p in self._flat_params
        ]
        self._static_sharpness = list(self._flat_sharpness)

        # Pre-compute static scalars (won't change during graph replay)
        self._static_alpha_mus = [
            float(max(0.0, min(1.0, self._cached_alpha * self._flat_layer_alphas[i])))
            for i in range(self._num_params)
        ]
        group = self.param_groups[0]
        beta1 = group["betas"][0]
        beta2 = group["betas"][1]
        self._static_beta2 = float(beta2)
        self._static_lr = float(group["lr"])
        self._static_eps = float(group["eps"])

        self._compile_prepared = True

    @torch.no_grad()
    def step_compiled(self, train_loss=None, val_loss=None, train_acc=None):
        """Graph-capturable eager step — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    @torch.no_grad()
    def sg2_step_megakernel(self, params, grads, sharpness_list, states, meta,
                            scalars):
        """SG2 eager optimizer-phase megakernel driver — REMOVED (pure L3-TC).

        The production L3-TC path runs SG2 via dispatch.fused_train_step (ops.
        sg2_fused_step), which fuses the REAL model fwd+bwd with this optimizer
        phase in ONE launch. This standalone eager optimizer-only driver is gone.
        """
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def _single_param_step(self, param, group, state):
        """Per-parameter eager step (use_grad_hooks path) — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def get_global_step(self):
        return self._global_step

    def get_cached_alpha(self):
        return self._cached_alpha

    def get_effective_wd(self):
        if self.param_groups:
            return self._get_effective_wd(self.param_groups[0]["weight_decay"])
        return 0.0

    # ── Checkpoint fidelity (same contract as SuperGrok11/15) ───────────────
    # All evolving state lives outside base Optimizer.state: the flat Adam
    # moments (+ INT8 scales under config3), the expert-EMA mu, the grokfast
    # slow EMA, sharpness, the per-coordinate GRU hidden states, per-param step
    # counters, the trained CSA/HCA/PEER meta-net (whose expert_counts /
    # step_counter recycling buffers ride its own state_dict), and the
    # step_full-internal meta Adam. Serialize all of it or a resume silently
    # resets the optimizer AND the trained meta-net. Derived constants
    # (_flat_layer_*) are rebuilt by __init__ and not serialized; the legacy
    # _flat_mamba_*_states are None placeholders and skipped.
    _EXTRA_KEY = "supergrok2_extra"
    _EXTRA_TENSOR_LISTS = ("flat_exp_avgs", "flat_exp_avg_sqs",
                           "flat_exp_avg_scales", "flat_mus", "flat_slows",
                           "flat_sharpness", "flat_gru_states")

    def state_dict(self):
        sd = super().state_dict()
        self._ensure_state()
        sd[self._EXTRA_KEY] = {
            "global_step": self._global_step,
            "step_counter": self._step_counter,
            "cached_alpha": self._cached_alpha,
            "cached_train_acc": self._cached_train_acc,
            "flat_steps": list(self._flat_steps),
            "meta_net": self.meta_net.state_dict(),
        }
        for key in self._EXTRA_TENSOR_LISTS:
            sd[self._EXTRA_KEY][key] = [t.detach().clone()
                                        for t in getattr(self, "_" + key)]
        if hasattr(self, "_auto_meta_opt"):
            sd[self._EXTRA_KEY]["auto_meta_opt"] = self._auto_meta_opt.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        extra = state_dict.get(self._EXTRA_KEY)
        base = {k: v for k, v in state_dict.items() if k != self._EXTRA_KEY}
        super().load_state_dict(base)
        if extra is None:
            return  # plain/base checkpoint: behave like the base class
        self._ensure_state()
        n = len(self._flat_params)
        for key in self._EXTRA_TENSOR_LISTS:
            saved = extra[key]
            live = getattr(self, "_" + key)
            if len(saved) != len(live):
                raise ValueError(
                    f"SuperGrok2.load_state_dict: checkpoint has {len(saved)} "
                    f"{key} buffers but the optimizer holds {len(live)} (model "
                    f"params: {n}) — architecture or state_precision changed.")
            for dst, src in zip(live, saved):
                if dst.numel() != src.numel():
                    raise ValueError(
                        f"SuperGrok2.load_state_dict: {key} numel mismatch "
                        f"({dst.numel()} vs checkpoint {src.numel()}).")
                dst.copy_(src.to(dst.device, dtype=dst.dtype))
        self._flat_steps = list(extra["flat_steps"])
        self._global_step = int(extra["global_step"])
        self._step_counter = int(extra["step_counter"])
        self._cached_alpha = float(extra["cached_alpha"])
        self._cached_train_acc = float(extra["cached_train_acc"])
        self.meta_net.load_state_dict(extra["meta_net"])
        if "auto_meta_opt" in extra:
            if not hasattr(self, "_auto_meta_opt"):
                self._auto_meta_opt = torch.optim.Adam(
                    self.meta_net.parameters(), lr=1e-4)
            self._auto_meta_opt.load_state_dict(extra["auto_meta_opt"])
        self._weights_dirty = True   # re-extract kernel weights from the loaded net

    def step_full(self, model, train_x, train_y, val_x, val_y, criterion=None):
        """Complete training step: forward + backward + SAM + meta-learning + optimizer."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if not hasattr(self, '_auto_meta_opt'):
            self._auto_meta_opt = torch.optim.Adam(self.meta_net.parameters(), lr=1e-4)

        model.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()

        metrics: Dict[str, float] = {}
        step_num = self._global_step + 1

        sam_freq_eff = self._get_effective_sam_freq()
        if step_num % sam_freq_eff == 0:
            try:
                metrics["sam_loss"] = self.sam_step(model, train_x, train_y, criterion)
            except RuntimeError as e:
                import warnings
                warnings.warn(f"SuperGrok2 SAM step failed at step {step_num}: {e}")

        bilevel_freq_eff = self._get_effective_bilevel_freq()
        if step_num % bilevel_freq_eff == 0:
            try:
                _vl = self.bilevel_step(
                    model, train_x, train_y, val_x, val_y, criterion, self._auto_meta_opt)
                metrics["val_loss"] = _vl.item() if torch.is_tensor(_vl) else _vl
            except RuntimeError as e:
                import warnings
                warnings.warn(f"SuperGrok2 bilevel step failed at step {step_num}: {e}")

        kw: Dict[str, float] = {}
        alpha_freq = self.alpha_update_freq
        if (step_num % alpha_freq == 0) or step_num == 1:
            with torch.no_grad():
                # Batch the GPU→CPU transfer: compute both metrics then sync once
                _tl = loss.detach()
                _ta = (logits.detach().argmax(-1) == train_y).float().mean()
                train_loss_val = _tl.item()
                train_acc = _ta.item()
            kw["train_loss"] = train_loss_val
            kw["train_acc"] = train_acc
            metrics["train_loss"] = train_loss_val
            metrics["train_acc"] = train_acc
            if step_num % alpha_freq == 0:
                with torch.no_grad():
                    val_loss_val = criterion(model(val_x), val_y).item()
                kw["val_loss"] = val_loss_val
                if "val_loss" not in metrics:
                    metrics["val_loss"] = val_loss_val

        self.step(**kw)
        return metrics


class CompiledSuperGrok2:
    """CUDA graph wrapper for SuperGrok2 with warmup-capture-replay cycle.

    Provides zero-overhead optimizer step replay after initial capture.
    Falls back to eager mode gracefully when CUDA graphs are unavailable.

    Usage::

        opt = SuperGrok2(model.parameters(), ...)
        compiled_opt = CompiledSuperGrok2(opt, warmup_steps=3)

        for step in range(n_steps):
            loss.backward()
            compiled_opt.step()  # Warmup → capture → replay automatically

    Args:
        optimizer: A :class:`SuperGrok2` instance.
        warmup_steps: Number of eager steps before graph capture (default: 3).
            Must be >= 1 to ensure optimizer state is initialized.
        enable_compile: Whether to attempt torch.compile on the step function
            (default: False). Requires PyTorch 2.0+.
        recycle_in_eager: Whether to periodically drop to eager mode for
            expert recycling (default: True). When True, every
            ``recycle_interval`` steps the graph is bypassed for one step.
    """

    def __init__(
        self,
        optimizer: SuperGrok2,
        warmup_steps: int = 3,
        enable_compile: bool = False,
        recycle_in_eager: bool = True,
    ):
        if not isinstance(optimizer, SuperGrok2):
            raise TypeError(f"Expected SuperGrok2, got {type(optimizer)}")

        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.enable_compile = enable_compile
        self.recycle_in_eager = recycle_in_eager

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._step_count = 0
        self._graph_valid = False
        self._compiled_step = None

        # Static gradient buffers (allocated during capture)
        self._static_grad_buffers: Dict[int, torch.Tensor] = {}

    def step(self, **kwargs):
        """Compiled-wrapper eager step — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def _capture_graph(self):
        """Capture the optimizer step as a CUDA graph."""
        try:
            if not torch.cuda.is_available():
                self._graph_valid = False
                return

            # Prepare static buffers
            self.optimizer._prepare_for_compile()
            self._allocate_static_grads()
            self._copy_grads_to_static()

            # Swap to static grads for capture
            orig_grads = self._swap_to_static_grads()

            # Optionally torch.compile the step function
            if self.enable_compile and self._compiled_step is None:
                try:
                    self._compiled_step = torch.compile(
                        self.optimizer.step_compiled, fullgraph=False)
                except Exception:
                    self._compiled_step = self.optimizer.step_compiled

            step_fn = self._compiled_step or self.optimizer.step_compiled

            # Record CUDA graph
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                step_fn()

            # Restore original grads
            self._restore_grads(orig_grads)
            self._graph_valid = True

        except Exception as exc:
            self._graph = None
            self._graph_valid = False
            # Warn rather than silently degrade: CUDA-graph capture failed, so
            # every subsequent step pays the eager-launch overhead this wrapper
            # exists to eliminate. Surfacing the cause lets the caller fix it
            # (e.g. dynamic shapes / sync ops in the captured region).
            warnings.warn(
                f"SuperGrok2 CUDA-graph capture failed; falling back to eager "
                f"steps (launch overhead not amortized). Cause: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fall back to eager
            self.optimizer.step()

    def _allocate_static_grads(self):
        """Allocate static gradient buffers mirroring parameter gradients."""
        self._static_grad_buffers.clear()
        for p in self.optimizer._flat_params:
            pid = id(p)
            if p.grad is not None:
                self._static_grad_buffers[pid] = torch.empty_like(p.grad)
            else:
                self._static_grad_buffers[pid] = torch.zeros(
                    p.data.numel(), dtype=torch.float32, device=p.device)

    def _copy_grads_to_static(self):
        """Copy current gradients into static buffers for graph replay."""
        for p in self.optimizer._flat_params:
            pid = id(p)
            if pid in self._static_grad_buffers and p.grad is not None:
                self._static_grad_buffers[pid].copy_(p.grad.data)

    def _swap_to_static_grads(self) -> Dict[int, Optional[torch.Tensor]]:
        """Replace parameter grads with static buffers, return originals."""
        orig = {}
        for p in self.optimizer._flat_params:
            pid = id(p)
            orig[pid] = p.grad
            if pid in self._static_grad_buffers:
                p.grad = self._static_grad_buffers[pid]
        return orig

    def _restore_grads(self, orig_grads: Dict[int, Optional[torch.Tensor]]):
        """Restore original parameter gradients."""
        for p in self.optimizer._flat_params:
            pid = id(p)
            if pid in orig_grads:
                p.grad = orig_grads[pid]

    def invalidate(self):
        """Force re-capture of the CUDA graph on the next step."""
        self._graph_valid = False
        self._graph = None

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def meta_net(self):
        return self.optimizer.meta_net

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self.invalidate()

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self.optimizer, name)


# ════════════════════════════════════════════════════════════════════════════
#  MoEAwareSuperGrok2 — MoE-aware variant (folded in from moe_adam.py)
#
#  Compacts active expert parameters before running the full SG2 metanet:
#  for top-k routing with E experts, the scan processes only k/E of expert
#  params instead of 100%. Counts expert activations for load balancing and
#  applies frequency-based per-expert LR scaling.
# ════════════════════════════════════════════════════════════════════════════


class MoEAwareSuperGrok2(SuperGrok2):
    """SuperGrok v2 with deep MoE optimization.

    Extends SuperGrok2 to be MoE-aware: when active_expert_indices are
    provided, filters to only active-gradient params, runs a shorter
    compacted scan, then scatters results back.
    """

    def __init__(self, params, lr=1e-3, moe_config=None, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.moe_config = moe_config or {}
        self._expert_counts = None
        self._lr_scale = None
        self._load_balance_coeff = self.moe_config.get('load_balance_coeff', 0.01)
        self._freq_smoothing = self.moe_config.get('freq_smoothing', 0.9)
        self._min_lr_scale = self.moe_config.get('min_lr_scale', 0.1)
        self._max_lr_scale = self.moe_config.get('max_lr_scale', 10.0)

    def step(self, closure=None, active_expert_indices=None,
             gate_logits=None, param_to_expert=None, expert_active=None,
             threshold=0.0, **kwargs):
        """MoE-aware eager step — REMOVED (pure L3-TC)."""
        raise NotImplementedError(
            "L3-TC megakernel only; eager .step() removed — the megakernel owns "
            "the optimizer update via fused_train_step")

    def _moe_step(self, active_expert_indices, gate_logits=None,
                  param_to_expert=None, expert_active=None,
                  threshold=0.0, closure=None, **kwargs):
        """MoE-aware step: compact -> attention meta-net -> scatter.

        The device-side MoE-model compaction kernels — ``moe_count_expert_
        activations``, ``moe_compute_load_balance_loss``, ``moe_apply_frequency_
        scaling``, ``moe_filter_active_params``, ``moe_scatter_results`` — are
        implemented for both sm_90 (real CUDA) and gfx942 (ATen) and were
        numerically reviewed (Stage 1B). This path is reached only when the C++
        extension is built and exposes them (guarded at the call site by
        ``_HAS_CUDA and hasattr(_ops, 'moe_filter_active_params')``); otherwise
        the caller transparently uses the dense ``SuperGrok2.step()``.

        NOTE (🟡): the kernels are compile-verified but their on-device numerics
        are hardware-deferred (see HARDWARE_VALIDATION.md Stage 1B). The dense
        path remains the default; pass ``active_expert_indices`` to opt in to the
        compacted expert path on real hardware.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._ensure_state()
        self._global_step += 1

        device = active_expert_indices.device
        num_experts = self.moe_config.get('num_experts', 64)

        # Initialize expert tracking state
        if self._expert_counts is None:
            self._expert_counts = torch.zeros(
                num_experts, dtype=torch.int32, device=device)
        if self._lr_scale is None:
            self._lr_scale = torch.ones(
                num_experts, dtype=torch.float32, device=device)

        # Step 1: Count expert activations for load balancing
        if gate_logits is not None:
            N_gate = gate_logits.shape[0]
            self._expert_counts.zero_()
            _ops.moe_count_expert_activations(
                gate_logits, self._expert_counts,
                threshold, N_gate, num_experts,
            )

            # Compute load balance auxiliary loss
            lb_loss = _ops.moe_compute_load_balance_loss(
                self._expert_counts, gate_logits, N_gate, num_experts,
            )
            self._cached_load_balance_loss = lb_loss.item() * self._load_balance_coeff

            # Update frequency-based LR scaling
            total_act = int(self._expert_counts.sum().item())
            _ops.moe_apply_frequency_scaling(
                self._expert_counts, self._lr_scale,
                num_experts, total_act,
                self._min_lr_scale, self._max_lr_scale, self._freq_smoothing,
            )

        # Step 2-4: Filter, compact scan, scatter — only if we have the mappings
        if param_to_expert is not None and expert_active is not None:
            for i, p in enumerate(self._flat_params):
                if p.grad is None:
                    continue
                total_params = p.grad.numel()
                max_active = total_params  # upper bound

                compact_params = torch.empty(max_active, device=device)
                compact_grads = torch.empty(max_active, device=device)
                compact_state_m = torch.empty(max_active, device=device)
                compact_state_v = torch.empty(max_active, device=device)
                scatter_indices = torch.empty(max_active, dtype=torch.int32, device=device)
                compact_count = torch.zeros(1, dtype=torch.int32, device=device)

                _ops.moe_filter_active_params(
                    p.data.reshape(-1), p.grad.data.reshape(-1),
                    self._flat_exp_avgs[i].reshape(-1),
                    self._flat_exp_avg_sqs[i].reshape(-1),
                    param_to_expert, expert_active,
                    compact_params, compact_grads,
                    compact_state_m, compact_state_v,
                    scatter_indices, compact_count,
                    total_params,
                )

                N_active = compact_count.item()
                if N_active > 0:
                    _ops.moe_scatter_results(
                        compact_params[:N_active],
                        compact_state_m[:N_active],
                        compact_state_v[:N_active],
                        scatter_indices[:N_active],
                        p.data.reshape(-1),
                        self._flat_exp_avgs[i].reshape(-1),
                        self._flat_exp_avg_sqs[i].reshape(-1),
                        N_active,
                    )

        # Fall through to standard step for non-expert params
        return super().step(**kwargs) if loss is None else loss

    @property
    def load_balance_loss(self) -> float:
        """Return the most recent load balance auxiliary loss."""
        return getattr(self, '_cached_load_balance_loss', 0.0)

    @property
    def expert_lr_scales(self) -> Optional[torch.Tensor]:
        """Return the current per-expert LR scale factors."""
        return self._lr_scale


# ── Shared (inlined) helper: register post_accumulate_grad_hook on each param.
# Each hook calls back into the optimizer's `_single_param_step` so the update
# runs while gradient data is still L2-warm. Duplicated across every optimizer
# file by design (self-containment); requires PyTorch >= 2.1.
def _register_grad_hooks(optimizer):
    # Fail-fast (construction time): the per-parameter eager `_single_param_step`
    # was removed under pure L3-TC (the megakernel owns the optimizer update via
    # fused_train_step), so the grad-hook path cannot run. Raise HERE — when the
    # optimizer is constructed with use_grad_hooks=True — rather than mid-backward
    # from inside the hook, so `--grad-hooks` fails immediately with a clear
    # message. The plumbing below is retained for when an eager path returns.
    raise NotImplementedError(
        "use_grad_hooks is not supported: the per-parameter eager step was "
        "removed under pure L3-TC (the megakernel owns the optimizer update via "
        "fused_train_step). Construct the optimizer with use_grad_hooks=False "
        "(the default) and drive training through the fused L3-TC megakernel.")
    _pt = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if _pt < (2, 1):
        raise RuntimeError(
            f"use_grad_hooks requires PyTorch >= 2.1 for "
            f"register_post_accumulate_grad_hook. Current: {torch.__version__}.")
    optimizer._grad_hook_handles = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue
            def _hook(param, _g=group, _opt=optimizer):
                if param.grad is None:
                    return
                _opt._single_param_step(param, _g, _opt.state[param])
            optimizer._grad_hook_handles.append(
                p.register_post_accumulate_grad_hook(_hook))
