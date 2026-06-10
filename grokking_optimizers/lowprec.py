"""H100-native low-precision matmul modes: FP8 (E4M3/E5M2) and INT8.

Owner directive: the precision analysis must cover everything ≤16-bit that is
native to H100 tensor cores. Unlike bf16/fp16 these are NOT autocast dtypes —
they are implemented here as nn.Linear swaps using the native kernels:

  • fp8      — standard hybrid recipe: E4M3 forward (weights+acts), E5M2
               gradients in backward, dynamic per-tensor scaling, via
               torch._scaled_mm (Hopper FP8 tensor cores).
  • fp8e5m2  — E5M2 forward variant (more range, less mantissa); same backward.
  • int8     — dynamic symmetric per-tensor quantization, forward GEMM via
               torch._int_mm (Hopper IMMA); backward runs in bf16 ("int8-fwd /
               bf16-bwd" — int8 gradient training is not a defined recipe;
               this is the honest standard semantics, stated loudly).

Master weights / optimizer state / gradients at the optimizer boundary remain
fp32 — identical contract to the bf16/tf32 modes. Non-Linear ops (attention
bmms, norms, scans) run under bf16 autocast in these modes (the standard
low-precision-training carrier).

ALIGNMENT FALLBACK (loud, never silent): _scaled_mm/_int_mm constrain GEMM
dims (e.g. K/N multiples of 16 for fp8). A layer that cannot run the native
kernel (e.g. the 99-logit head, vit's 49-dim patch embed) is left as a
plain Linear (bf16 via autocast) and COUNTED; `swap_report()` exposes exactly
which layers run native vs fallback so no run can misreport its precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["swap_linears_lowprec", "LOWPREC_MODES"]

LOWPREC_MODES = ("fp8", "fp8e5m2", "int8")

_E4M3 = torch.float8_e4m3fn
_E5M2 = torch.float8_e5m2


def _amax_scale(t: torch.Tensor, fp8_dtype) -> torch.Tensor:
    """Per-tensor dynamic scale mapping amax -> fp8 max representable."""
    fmax = torch.finfo(fp8_dtype).max
    amax = t.abs().amax().clamp(min=1e-12)
    return (fmax / amax).to(torch.float32)


def _scaled_mm_compat(a8, b8, scale_a, scale_b, out_dtype):
    """torch._scaled_mm across torch versions (tuple return in <=2.3)."""
    out = torch._scaled_mm(a8, b8, scale_a=scale_a, scale_b=scale_b,
                           out_dtype=out_dtype)
    if isinstance(out, tuple):
        out = out[0]
    return out


class _FP8MatmulFn(torch.autograd.Function):
    """y = x @ W^T (+b) with fp8 tensor-core GEMMs.

    Forward in `fwd_dtype` (E4M3 standard / E5M2 variant); backward computes
    dX and dW with E5M2 gradients (the standard recipe). Scales are dynamic
    per-tensor (recomputed each call — correctness over caching).
    """

    @staticmethod
    def forward(ctx, x, w, bias, fwd_dtype):
        x2 = x.reshape(-1, x.shape[-1])
        # _scaled_mm requires row-major @ column-major. NOTE: sm_90 supports
        # e4m3×e4m3 and MIXED e5m2×e4m3 pairings but NOT e5m2×e5m2 — so the
        # "fp8e5m2" variant means E5M2 ACTIVATIONS × E4M3 weights.
        sx = 1.0 / _amax_scale(x2, fwd_dtype)
        sw = 1.0 / _amax_scale(w, _E4M3)
        x8 = (x2 * (1.0 / sx)).to(fwd_dtype)
        w8 = (w * (1.0 / sw)).to(_E4M3)
        y = _scaled_mm_compat(x8, w8.t(), sx, sw, torch.bfloat16)
        if bias is not None:
            y = y + bias.to(y.dtype)
        ctx.save_for_backward(x2, w)
        ctx.x_shape = x.shape
        return y.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, w = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1])
        sg = 1.0 / _amax_scale(g2, _E5M2)
        g8 = (g2 * (1.0 / sg)).to(_E5M2)
        # dX = g @ W : [T,out]@[out,in] — mat2 must be COLUMN-major [out,in];
        # w8d is row-major [out,in] → .t().contiguous().t() re-lays it
        # column-major with the same logical shape. (E5M2 × E4M3 is a
        # supported sm_90 pairing.)
        swd = 1.0 / _amax_scale(w, _E4M3)
        w8d = (w * (1.0 / swd)).to(_E4M3)
        dx = _scaled_mm_compat(g8, w8d.t().contiguous().t(), sg, swd,
                               torch.bfloat16)
        # dW = g^T @ x : [out,T]@[T,in] — here K = T (token count), which is
        # not generally %16 (e.g. 4191×4=16764). Zero-pad T up to %16 for this
        # GEMM only: zero rows contribute exactly nothing to gᵀ@x (exact, and
        # amax/scales are unaffected by added zeros).
        sxd = 1.0 / _amax_scale(x2, _E4M3)
        T = g2.shape[0]
        Tp = (T + 15) // 16 * 16
        if Tp != T:
            pad = Tp - T
            g2p = torch.nn.functional.pad(g2, (0, 0, 0, pad))
            x2p = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        else:
            g2p, x2p = g2, x2
        g8p = (g2p * (1.0 / sg)).to(_E5M2)
        x8d = (x2p * (1.0 / sxd)).to(_E4M3)
        gt8 = g8p.t().contiguous()
        dw = _scaled_mm_compat(gt8, x8d.t().contiguous().t(), sg, sxd,
                               torch.bfloat16)
        db = gy.reshape(-1, gy.shape[-1]).sum(0) if ctx.needs_input_grad[2] else None
        return (dx.reshape(ctx.x_shape).to(torch.float32),
                dw.to(torch.float32), db, None)


class FP8Linear(nn.Module):
    def __init__(self, base: nn.Linear, fwd_dtype):
        super().__init__()
        self.weight = base.weight        # fp32 master (shared param object)
        self.bias = base.bias
        self.fwd_dtype = fwd_dtype

    def forward(self, x):
        return _FP8MatmulFn.apply(x.to(torch.float32), self.weight, self.bias,
                                  self.fwd_dtype)


class _Int8MatmulFn(torch.autograd.Function):
    """Forward: dynamic symmetric int8 GEMM via torch._int_mm (IMMA cores).
    Backward: bf16 mats (int8 gradients are not a defined training recipe)."""

    @staticmethod
    def forward(ctx, x, w, bias):
        x2 = x.reshape(-1, x.shape[-1])
        sx = x2.abs().amax().clamp(min=1e-12) / 127.0
        sw = w.abs().amax().clamp(min=1e-12) / 127.0
        # cublasLt IMMA needs the token dim aligned (the race batch, e.g.
        # 4191×4=16764, is not %8) — zero-pad rows for the GEMM (exact: zero
        # rows yield zero outputs) and slice them back off.
        T = x2.shape[0]
        Tp = (T + 15) // 16 * 16
        x2p = torch.nn.functional.pad(x2, (0, 0, 0, Tp - T)) if Tp != T else x2
        xq = torch.clamp((x2p / sx).round(), -127, 127).to(torch.int8)
        wq = torch.clamp((w / sw).round(), -127, 127).to(torch.int8)
        # mat2 must be a COLUMN-major [K,N] view (wq is [out,in] row-major →
        # .t() is exactly that); forcing .contiguous() here flips it row-major
        # and cublasLt IMMA rejects the layout (CUBLAS_STATUS_NOT_SUPPORTED).
        y32 = torch._int_mm(xq, wq.t())
        y = y32[:T].to(torch.float32) * (sx * sw)
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x2, w)
        ctx.x_shape = x.shape
        return y.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, w = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1]).to(torch.bfloat16)
        dx = (g2 @ w.to(torch.bfloat16)).to(torch.float32)
        dw = (g2.t() @ x2.to(torch.bfloat16)).to(torch.float32)
        db = gy.reshape(-1, gy.shape[-1]).sum(0) if ctx.needs_input_grad[2] else None
        return dx.reshape(ctx.x_shape), dw, db


class Int8Linear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias

    def forward(self, x):
        return _Int8MatmulFn.apply(x.to(torch.float32), self.weight, self.bias)


def _fp8_dims_ok(in_f: int, out_f: int) -> bool:
    # _scaled_mm: K and N must be multiples of 16 on sm_90
    return in_f % 16 == 0 and out_f % 16 == 0


def _int8_dims_ok(in_f: int, out_f: int) -> bool:
    # _int_mm: K %8; result dims constraints are looser, probe conservatively
    return in_f % 8 == 0 and out_f % 8 == 0


def swap_linears_lowprec(model: nn.Module, mode: str):
    """Swap every compatible nn.Linear for the native low-precision version.

    Returns a report dict {"mode", "swapped": [...], "fallback_bf16": [...]}.
    The fallback list is the loud record of layers that CANNOT run the native
    kernel (dim alignment) and therefore stay plain Linear under the bf16
    autocast carrier — never silent.
    """
    assert mode in LOWPREC_MODES, mode
    swapped, fallback = [], []

    def _convert(parent, name, child):
        full = f"{parent}.{name}" if parent else name
        if mode in ("fp8", "fp8e5m2"):
            if _fp8_dims_ok(child.in_features, child.out_features):
                fwd = _E4M3 if mode == "fp8" else _E5M2
                return FP8Linear(child, fwd), full, True
            return child, full, False
        else:  # int8
            if _int8_dims_ok(child.in_features, child.out_features):
                return Int8Linear(child), full, True
            return child, full, False

    def _walk(mod, prefix=""):
        for name, child in list(mod.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                new, fname, ok = _convert(prefix, name, child)
                if ok:
                    setattr(mod, name, new)
                    swapped.append(f"{fname}[{child.in_features}x{child.out_features}]")
                else:
                    fallback.append(f"{fname}[{child.in_features}x{child.out_features}]")
            else:
                _walk(child, full)

    _walk(model)
    report = {"mode": mode, "swapped": swapped, "fallback_bf16": fallback}
    import warnings
    if not swapped:
        warnings.warn(f"lowprec mode={mode}: NO layer was compatible — the run "
                      f"is effectively bf16. fallbacks={fallback}")
    elif fallback:
        warnings.warn(f"lowprec mode={mode}: {len(swapped)} layers native, "
                      f"{len(fallback)} stayed bf16 (dim alignment): {fallback}")
    return report
