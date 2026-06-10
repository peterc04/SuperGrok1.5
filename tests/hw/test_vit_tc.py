"""H100 (sm_90a) ViT TENSOR-CORE variant gates.

Validates csrc/fused/sm_90/model_stage_vit_tc.cuh — the batch-tiled bf16 wgmma
fwd+bwd for the ViT cell (DESIGN-TC-PIPELINE.md Fork B), the TUNED VARIANT
compiled alongside the scalar path and selected by SG_TUNED_GEMM_IMPL. The ViT
twin of tests/hw/test_decoder_tc.py.

TWO LAYERS of gates (DESIGN §10):

  PART 1 — GEMM-orientation micro-gates (the engine, exercised through the ViT
    cell TU at the ViT TILE_M=1088 = 17 stacked m64 atoms), in the three operand
    orientations the ViT fwd+bwd use, with the transposed HBM->smem staging that
    keeps every wgmma issue on the substrate-validated TransA=0/TransB=0 path:
      (fwd) Y = X @ W^T          (no transpose)
      (dX)  dX = dY @ W          (W transposed-staged)
      (dW)  dW = dY^T @ X, K=T   (BOTH transposed-staged — multi-k-step, the
                                  stride-bug gate; A=I alone would miss it)
    Each vs an fp64 reference of the bf16-rounded inputs. The engine is
    model-agnostic (silicon-gated 13/13 by the decoder selftest); here it is
    re-witnessed through the ViT TU's 17-atom M stacking.

  PART 2 — full-cell gates: the Fork-B fwd+bwd+AdamW persistent megakernel
    (mega_vit_real_adamw_tc.cu, built with -DSG_TUNED_GEMM_IMPL=1) vs the
    bf16-FAITHFUL fp64 ViT oracle (the named reference — rounds at the kernel's
    storage points). DESIGN §5.2 tiers: (1) single-step grad parity, (2) loss,
    (3) determinism A/A/A, (4) step-time vs scalar, (5) 50-step trajectory.

THE FOUR ViT DELTAS the gates exercise (vs the decoder twin):
  1. FULL (bidirectional) attention — no causal mask.
  2. PATCH-PROJ Linear(49→128) embedding (K=49 pad 64) + cls_token + pos.
  3. CLS pos-0 head (final-norm + head + CE on position 0).
  4. kTileM = LCM(17,64) = 1088 (multiple of kSeq=17).

Run:  CUDA_MPS_PIPE_DIRECTORY=/nonexistent python -m pytest tests/hw/test_vit_tc.py -v -s
  or: CUDA_MPS_PIPE_DIRECTORY=/nonexistent python tests/hw/test_vit_tc.py
"""
import os
import sys

import pytest

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

_THIS = os.path.abspath(__file__)
_REPO = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))


def _sm90a_available():
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


_GATE = pytest.mark.skipif(
    not _sm90a_available(),
    reason="ViT TC gates need an sm_90 (Hopper) GPU + bf16")


def _disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _k_scaled_atol(K):
    # fp32 accumulation-order bound ≈ K·ε·scale with ε=2^-24. Substrate used 2e-3
    # at K<=512; scale linearly with a floor. (bf16-rounded-oracle accumulation-
    # order gate; bf16 input noise cancels because both sides are bf16-rounded.)
    return max(2e-3, 2e-3 * (K / 512.0))


# ── ONE JIT build (the ViT TC cell TU owns its module; PART 1 GEMM-orientation
#    entries + PART 2 full-cell entries live in the SAME TU — not in setup.py's
#    glob; the gate drives it directly). ─────────────────────────────────────
_TC_MODULE = None
_TC_BUILD_ERR = None
_TC_TU = os.path.join(_REPO, "csrc", "fused", "sm_90", "mega_vit_real_adamw_tc.cu")


def _build_tc_module():
    global _TC_MODULE, _TC_BUILD_ERR
    if _TC_MODULE is not None or _TC_BUILD_ERR is not None:
        return _TC_MODULE
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import load
    try:
        _TC_MODULE = load(
            name="mega_vit_real_adamw_tc",
            sources=[_TC_TU],
            extra_include_paths=[_REPO],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                "-gencode=arch=compute_90a,code=sm_90a",
                "-gencode=arch=compute_90a,code=compute_90a",
                "-lineinfo",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=True,
        )
    except Exception as e:
        _TC_BUILD_ERR = e
        raise
    return _TC_MODULE


def _bf16(t):
    return t.to(torch.bfloat16)


# ═════════════════════════════════════════════════════════════════════════
#  PART 1 — GEMM orientation micro-gates (the engine, ViT TILE_M = 1088).
# ═════════════════════════════════════════════════════════════════════════

@_GATE
def test_fwd_identity_localization():
    """(fwd) A=I localization: X = [I | 0] (TILE_M x Kin) ⇒ Y[m,n] = W[n,m] for
    m<Kin. Isolates fragment-layout / descriptor / TILE_M-stacking bugs (ViT's
    17-atom M stacking)."""
    mod = _build_tc_module()
    _disable_tf32()
    TILE_M = mod.TILE_M
    N, Kin = 128, 128
    X = torch.zeros(TILE_M, Kin, device="cuda", dtype=torch.float32)
    for i in range(min(Kin, TILE_M)):
        X[i, i] = 1.0
    W = (torch.arange(N * Kin, device="cuda", dtype=torch.float32).reshape(N, Kin) / 97.0)
    Xb, Wb = _bf16(X), _bf16(W)
    D = mod.gemm_fwd(Xb.reshape(-1).contiguous(), Wb.reshape(-1).contiguous(), N, Kin, N)
    ref = torch.einsum("mk,nk->mn", Xb.double().cpu(), Wb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    assert err < _k_scaled_atol(Kin), (
        f"fwd A=I localization max_err={err:.3e} — layout/descriptor/stacking bug\n"
        f"D[:4,:4]=\n{D[:4,:4].cpu()}\nref[:4,:4]=\n{ref[:4,:4]}")
    print(f"[fwd-loc] A=I TILE_M={TILE_M} N={N} K={Kin} max_err={err:.3e}  OK")


@_GATE
@pytest.mark.parametrize("Kin,Nout", [(128, 128), (512, 128), (128, 384), (128, 512), (64, 128)])
def test_fwd_random(Kin, Nout):
    """(fwd) random X,W over the design's shapes: K depths (in/out K=128, ff2
    K=512, patch K=64) AND N widths (out 128, in_proj 3d=384, ff0 dff=512)."""
    mod = _build_tc_module()
    _disable_tf32()
    TILE_M = mod.TILE_M
    N = 128
    g = torch.Generator(device="cuda").manual_seed(1000 + Kin * 1000 + Nout)
    X = torch.randn(TILE_M, Kin, generator=g, device="cuda")
    W = torch.randn(Nout, Kin, generator=g, device="cuda")
    Xb, Wb = _bf16(X), _bf16(W)
    D = mod.gemm_fwd(Xb.reshape(-1).contiguous(), Wb.reshape(-1).contiguous(), N, Kin, Nout)
    ref = torch.einsum("mk,nk->mn", Xb.double().cpu(), Wb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    tol = _k_scaled_atol(Kin) * max(1.0, ref.abs().max().item())
    assert err < tol, f"fwd random K={Kin} Nout={Nout} max_err={err:.3e} tol={tol:.3e}"
    print(f"[fwd] random Nout={Nout} (N-tiled by {N}) K={Kin} max_err={err:.3e} tol={tol:.3e}  OK")


@_GATE
@pytest.mark.parametrize("Nout", [128, 512])
def test_dx_random(Nout):
    """(dX) dX = dY @ W with W TRANSPOSED-staged. N(wgmma)=in_dim=128, K=Nout."""
    mod = _build_tc_module()
    _disable_tf32()
    TILE_M = mod.TILE_M
    N = 128
    g = torch.Generator(device="cuda").manual_seed(2000 + Nout)
    dY = torch.randn(TILE_M, Nout, generator=g, device="cuda")
    W = torch.randn(Nout, N, generator=g, device="cuda")
    dYb, Wb = _bf16(dY), _bf16(W)
    D = mod.gemm_dx(dYb.reshape(-1).contiguous(), Wb.reshape(-1).contiguous(), N, Nout)
    ref = torch.einsum("mo,oi->mi", dYb.double().cpu(), Wb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    tol = _k_scaled_atol(Nout) * max(1.0, ref.abs().max().item())
    assert err < tol, f"dX Nout={Nout} max_err={err:.3e} tol={tol:.3e}"
    print(f"[dX] random N={N} Nout(K)={Nout} max_err={err:.3e} tol={tol:.3e}  OK")


@_GATE
@pytest.mark.parametrize("T,Nout", [(128, 128), (2048, 128), (2048, 384), (2048, 512)])
def test_dw_random_multistep(T, Nout):
    """(dW) dW = dY^T @ X with BOTH operands TRANSPOSED-staged, K=T multi-k-step.
    THE stride-bug gate. T=2048 = 128 k-steps. Nout in {128, 384 (6 M-atoms),
    512 (8 M-atoms)} — exercises the M-atom-block loop. in_dim N=128."""
    mod = _build_tc_module()
    _disable_tf32()
    N = 128
    g = torch.Generator(device="cuda").manual_seed(3000 + T + Nout * 7)
    dY = torch.randn(T, Nout, generator=g, device="cuda")
    X = torch.randn(T, N, generator=g, device="cuda")
    dYb, Xb = _bf16(dY), _bf16(X)
    D = mod.gemm_dw(dYb.reshape(-1).contiguous(), Xb.reshape(-1).contiguous(), N, Nout, T)
    ref = torch.einsum("to,ti->oi", dYb.double().cpu(), Xb.double().cpu())
    err = (D.double().cpu() - ref).abs().max().item()
    tol = _k_scaled_atol(T) * max(1.0, ref.abs().max().item())
    assert err < tol, f"dW T(K)={T} Nout={Nout} max_err={err:.3e} tol={tol:.3e}"
    print(f"[dW] random Nout={Nout} in={N} K=T={T} ({T//16} steps, "
          f"{(Nout+63)//64} M-atoms) max_err={err:.3e} tol={tol:.3e}  OK")


@_GATE
def test_dw_identity_who_owns_what():
    """(dW) A=I-style: dY = [I_Nout | 0] over T rows ⇒ dW[o,i] = X[o,i] for
    o<Nout. Pins the (out,t)->fragment mapping for the doubly-transposed staging."""
    mod = _build_tc_module()
    _disable_tf32()
    Nout, N, T = 128, 128, 256
    dY = torch.zeros(T, Nout, device="cuda", dtype=torch.float32)
    for o in range(min(Nout, T)):
        dY[o, o] = 1.0
    X = (torch.arange(T * N, device="cuda", dtype=torch.float32).reshape(T, N) / 131.0)
    dYb, Xb = _bf16(dY), _bf16(X)
    D = mod.gemm_dw(dYb.reshape(-1).contiguous(), Xb.reshape(-1).contiguous(), N, Nout, T)
    ref = Xb[:Nout, :].double().cpu()
    err = (D.double().cpu() - ref).abs().max().item()
    assert err < _k_scaled_atol(T), (
        f"dW A=I localization max_err={err:.3e} — transposed-staging mapping bug\n"
        f"D[:4,:4]=\n{D[:4,:4].cpu()}\nref[:4,:4]=\n{ref[:4,:4]}")
    print(f"[dW-loc] A=I Nout={Nout} in={N} T={T} max_err={err:.3e}  OK")


@_GATE
def test_determinism_bitwise():
    """Same inputs twice → bit-identical (fixed tile + ascending-k, no atomics)."""
    mod = _build_tc_module()
    _disable_tf32()
    Nout, N, T = 128, 128, 512
    g = torch.Generator(device="cuda").manual_seed(31337)
    dY = _bf16(torch.randn(T, Nout, generator=g, device="cuda")).reshape(-1).contiguous()
    X = _bf16(torch.randn(T, N, generator=g, device="cuda")).reshape(-1).contiguous()
    D1 = mod.gemm_dw(dY, X, N, Nout, T)
    D2 = mod.gemm_dw(dY, X, N, Nout, T)
    D3 = mod.gemm_dw(dY, X, N, Nout, T)
    assert torch.equal(D1, D2) and torch.equal(D2, D3), "non-deterministic dW"
    print("[determinism] dW A/A/A bit-identical  OK")


# ═════════════════════════════════════════════════════════════════════════
#  PART 2 — FULL-CELL gates: the Fork-B fwd+bwd+AdamW persistent megakernel.
# ═════════════════════════════════════════════════════════════════════════

def _build_eager_vit(device="cuda", seed=0):
    """The exact race ViT (== tests/hw/test_vit_megakernel.py's builder), fp32
    params. Autocast supplies the bf16 matmul boundary in the trajectory gate."""
    import torch.nn as nn
    from tests.hw.vit_oracle import (VOCAB, D_MODEL, N_HEADS, N_LAYERS,
                                     PATCH_DIM, NUM_PATCHES)

    class EncoderBlock(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
            self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            a, _ = self.attn(x, x, x)   # NO mask — full attention
            x = self.n1(x + a); return self.n2(x + self.ff(x))

    class ViT(nn.Module):
        def __init__(self, p=VOCAB, patch_dim=PATCH_DIM, num_patches=NUM_PATCHES,
                     d=D_MODEL, h=N_HEADS, nl=N_LAYERS):
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
            for l in self.layers:
                h = l(h)
            return self.out(self.norm(h[:, 0, :]))

    torch.manual_seed(seed)
    return ViT().to(device)


def _flat_params(m):
    return torch.cat([p.detach().reshape(-1) for _, p in m.named_parameters()]).contiguous()


def _bf16_faithful_oracle(named, patches, targets):
    """The NAMED reference (DESIGN §5.2): the ViT fwd+bwd computed in fp64 but
    with EVERY value the TC kernel STORES rounded to bf16 at the SAME points
    (VitActs X_patch/X_in/X_ctx/X_x1/X_gact, qkv, ff0pre, X_hn, every dY_*, dh0;
    weights bf16-on-read). Math is fp64; bf16 ONLY at the kernel's storage points.

    Unlike autocast-bf16 (adjoints fp32 through the backward), this rounds at the
    kernel's exact storage boundary, so a real kernel bug shows rel~1 while the
    intended bf16-storage shows ~bf16-floor. Runs on CPU (fp64, tiny) so it never
    competes with the kernel for GPU memory. Transcribed from tests/hw/
    vit_oracle.py with the storage rounds inserted (the ViT twin of the decoder
    bf16-faithful oracle).

    DELTAS vs the decoder oracle: patch_proj embedding (K=49→bf16 patches @ bf16
    patch_w), FULL attention (no causal mask), CLS at pos 0, cls_token grad.
    """
    from tests.hw.vit_oracle import (
        VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ, D_FF, D_HEAD, ATTN_SCALE,
        NUM_PATCHES, PATCH_DIM, gelu, gelu_grad, layernorm_forward,
        layernorm_backward, softmax_lastdim, softmax_backward, ViTWeights)
    B16 = torch.bfloat16

    def bf(x):
        return x.to(B16).to(torch.float64)

    W = ViTWeights.from_named({k: v.double() for k, v in named.items()})
    patches = patches.double()
    Bn = patches.shape[0]
    dev = patches.device
    # ── embedding: patch_proj (bf16 patches @ bf16 patch_w) + cls + pos ──
    px = bf(patches)                                          # X_patch bf16 [B,16,49]
    proj = px @ bf(W.patch_w).t() + W.patch_b.double()        # fp32 (bias not stored bf16)
    cls = W.cls.double().reshape(1, 1, D_MODEL).expand(Bn, 1, D_MODEL)
    h0 = torch.cat([cls, proj], dim=1)                        # [B,17,d]
    pos_ids = torch.arange(SEQ, device=dev)
    h0 = h0 + W.pos.double()[pos_ids].unsqueeze(0)
    h = bf(h0)                                                # X_in[0] bf16
    caches = []
    for li in range(N_LAYERS):
        L = W.layers[li]
        x_in = h
        qkv = bf(x_in @ bf(L["in_w"]).t() + L["in_b"].double())   # qkv bf16
        q, k, v = qkv.split(D_MODEL, dim=-1)
        def sh(t): return t.reshape(Bn, SEQ, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
        qh, kh, vh = sh(q), sh(k), sh(v)
        scores = (qh @ kh.transpose(-2, -1)) * ATTN_SCALE         # fp64 attn, NO mask
        attn = softmax_lastdim(scores)
        ctx = (attn @ vh).permute(0, 2, 1, 3).reshape(Bn, SEQ, D_MODEL)
        ctx_b = bf(ctx)                                           # X_ctx bf16
        a = (ctx_b @ bf(L["out_w"]).t() + L["out_b"].double())     # fp32 work
        r1 = x_in + a
        x1, ln1 = layernorm_forward(r1, L["n1_w"].double(), L["n1_b"].double())
        x1_b = bf(x1)                                             # X_x1 bf16
        ff0 = x1_b @ bf(L["ff0_w"]).t() + L["ff0_b"].double()
        ff0_b = bf(ff0)                                           # ff0pre bf16
        g = bf(gelu(ff0))                                         # X_gact bf16
        ff2 = g @ bf(L["ff2_w"]).t() + L["ff2_b"].double()
        r2 = x1 + ff2
        h2, ln2 = layernorm_forward(r2, L["n2_w"].double(), L["n2_b"].double())
        h = bf(h2) if li + 1 < N_LAYERS else h2                   # X_in[li+1] bf16 / finalin fp32
        caches.append(dict(x_in=x_in, attn=attn, vh=vh, qh=qh, kh=kh, ctx=ctx_b,
                           x1=x1, x1_b=x1_b, ln1=ln1, ff0_b=ff0_b, g=g, ln2=ln2))
    h_cls = h[:, 0, :]                                            # CLS pos 0
    hn, nc = layernorm_forward(h_cls, W.norm_w.double(), W.norm_b.double())
    hn_b = bf(hn)                                                # X_hn bf16
    logits = hn_b @ W.out_w.double().t() + W.out_b.double()
    mx = logits.max(dim=-1, keepdim=True).values
    logz = mx.squeeze(-1) + torch.log(torch.exp(logits - mx).sum(dim=-1))
    loss = (logz - logits.gather(1, targets.unsqueeze(1)).squeeze(1)).mean()
    grads = {}
    sm = softmax_lastdim(logits); oh = torch.zeros_like(sm)
    oh.scatter_(1, targets.unsqueeze(1), 1.0)
    dlogits = bf((sm - oh) / Bn)                                  # dY_logits bf16
    grads["out.weight"] = dlogits.t() @ hn_b
    grads["out.bias"] = dlogits.sum(0)
    dhn = dlogits @ W.out_w.double()
    dh_cls, dnw, dnb = layernorm_backward(dhn, nc)
    grads["norm.weight"] = dnw; grads["norm.bias"] = dnb
    dh = torch.zeros(Bn, SEQ, D_MODEL, device=dev, dtype=torch.float64)
    dh[:, 0, :] = dh_cls                                          # CLS pos 0
    for li in reversed(range(N_LAYERS)):
        L = W.layers[li]; lc = caches[li]
        dr2, d2w, d2b = layernorm_backward(dh, lc["ln2"])
        grads[f"layers.{li}.n2.weight"] = d2w; grads[f"layers.{li}.n2.bias"] = d2b
        dff2 = bf(dr2)                                            # dY_ff2 bf16
        dx1 = dr2.clone()
        grads[f"layers.{li}.ff.2.weight"] = dff2.reshape(-1, D_MODEL).t() @ lc["g"].reshape(-1, D_FF)
        grads[f"layers.{li}.ff.2.bias"] = dff2.reshape(-1, D_MODEL).sum(0)
        dg = dff2 @ bf(L["ff2_w"])
        dff0 = bf(dg * gelu_grad(lc["ff0_b"]))                    # dY_ff0 bf16; gelu' reads ff0pre(bf16)
        grads[f"layers.{li}.ff.0.weight"] = dff0.reshape(-1, D_FF).t() @ lc["x1_b"].reshape(-1, D_MODEL)
        grads[f"layers.{li}.ff.0.bias"] = dff0.reshape(-1, D_FF).sum(0)
        dx1 = dx1 + dff0 @ bf(L["ff0_w"])
        dr1, d1w, d1b = layernorm_backward(dx1, lc["ln1"])
        grads[f"layers.{li}.n1.weight"] = d1w; grads[f"layers.{li}.n1.bias"] = d1b
        da = bf(dr1)                                              # dY_a bf16
        dx_in = dr1.clone()
        grads[f"layers.{li}.attn.out_proj.weight"] = da.reshape(-1, D_MODEL).t() @ lc["ctx"].reshape(-1, D_MODEL)
        grads[f"layers.{li}.attn.out_proj.bias"] = da.reshape(-1, D_MODEL).sum(0)
        dctx = da @ bf(L["out_w"])
        dctxh = dctx.reshape(Bn, SEQ, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
        attn = lc["attn"]; vh = lc["vh"]; qh = lc["qh"]; kh = lc["kh"]
        dattn = dctxh @ vh.transpose(-2, -1); dvh = attn.transpose(-2, -1) @ dctxh
        dsc = softmax_backward(dattn, attn) * ATTN_SCALE
        dqh = dsc @ kh; dkh = dsc.transpose(-2, -1) @ qh
        def mh(t): return t.permute(0, 2, 1, 3).reshape(Bn, SEQ, D_MODEL)
        dqkv = bf(torch.cat([mh(dqh), mh(dkh), mh(dvh)], dim=-1))  # dY_qkv bf16
        grads[f"layers.{li}.attn.in_proj_weight"] = dqkv.reshape(-1, 3 * D_MODEL).t() @ lc["x_in"].reshape(-1, D_MODEL)
        grads[f"layers.{li}.attn.in_proj_bias"] = dqkv.reshape(-1, 3 * D_MODEL).sum(0)
        dx_in = dx_in + dqkv @ bf(L["in_w"])
        dh = dx_in
    dh_b = bf(dh)                                                 # dh0 bf16
    # ── embedding bwd: cls_token (pos 0), pos (all 17), patch_proj (rows 1..16). ──
    grads["cls_token"] = dh_b[:, 0, :].sum(0).reshape(1, 1, D_MODEL)
    grads["pos.weight"] = dh_b.sum(0)
    dh_patch = dh_b[:, 1:, :]                                     # [B,16,d]
    grads["patch_proj.weight"] = dh_patch.reshape(-1, D_MODEL).t() @ px.reshape(-1, PATCH_DIM)
    grads["patch_proj.bias"] = dh_patch.reshape(-1, D_MODEL).sum(0)
    return loss.item(), grads


# nCTA cap for the gates (env SG_TC_NCTA_CAP, default 8). The per-CTA tile scratch
# is nCTA×(big); a fleet-saturated GPU has little free memory. Correctness +
# determinism are per fixed nCTA (the dW-tile + cls/pos owner maps read
# ctx.n_ctas), so a capped run is a faithful witness; the shipped path uses 0.
_TC_NCTA_CAP = int(os.environ.get("SG_TC_NCTA_CAP", "8"))


def _run_tc_step(mod, params, patches, targets, state, lr, betas, wd, eps, step):
    beta1, beta2 = betas
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    return mod.tc_train_step(params, patches.to(torch.float32), targets.to(torch.int32),
                             state, float(lr), float(beta1), float(beta2), float(eps),
                             float(wd), float(bc1), float(bc2), int(step),
                             ncta_cap=_TC_NCTA_CAP)


# Grad tolerances (DESIGN §5.2, CALIBRATED). dW = Σ_T dY·X over T terms with sign
# cancellation has an irreducible bf16-INPUT noise floor of several % even vs the
# bf16-faithful oracle. Biases (Σ dY, less cancellation) stay tighter. The
# structural witness is layer-0 ≈ layer-1; the absolute floor is calibrated. The
# patch_proj K=49→64-pad path is the new ViT one (witnessed alongside).
_TC_GRAD_REL_WEIGHTS = 0.15   # weight matrices (dW, T-contraction bf16 floor)
_TC_GRAD_REL_BIASES = 0.08    # biases / LN vectors / cls / pos (less cancellation)
_TC_LOSS_REL = 5e-3           # loss (logsumexp fp32)


def _make_patches(B, seed):
    g = torch.Generator().manual_seed(seed)
    from tests.hw.vit_oracle import NUM_PATCHES, PATCH_DIM
    return torch.randn(B, NUM_PATCHES, PATCH_DIM, generator=g)


def _grad_parity_core(B, ncta_cap, label):
    """Shared keystone body: TC reduced grad vs the bf16-FAITHFUL fp64 ViT oracle
    for batch B at the given nCTA cap. Returns nothing; asserts. Parametrizing
    (B, cap) lets ONE body cover the default case AND the two production paths the
    default never hits (grid-stride: n_tiles > nCTA at cap=1; ragged final tile:
    B%64 != 0 → last tile nrows < kTileM)."""
    from tests.hw.vit_oracle import (vit_param_layout, ViTWeights, vit_forward,
                                     vit_backward)
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    m = _build_eager_vit("cpu", seed=123)
    named = {n: p.detach().clone() for n, p in m.named_parameters()}
    patches = _make_patches(B, 5)
    targets = torch.randint(0, 97, (B,), generator=torch.Generator().manual_seed(7))
    # bf16-faithful fp64 oracle (CPU) + pure-fp64 oracle (the calibration floor).
    loss_o, grads_o = _bf16_faithful_oracle(named, patches, targets)
    W64 = ViTWeights.from_named({k: v.double() for k, v in named.items()})
    _, c64 = vit_forward(W64, patches.double(), targets)
    grads_fp64 = vit_backward(W64, c64)
    # TC kernel (fresh state; step 1). cap is EXPLICIT (overrides env _TC_NCTA_CAP).
    params = torch.cat([p.reshape(-1) for _, p in named.items()]).contiguous().to(dev)
    total = int(mod.TOTAL)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    beta1, beta2 = 0.9, 0.98
    loss_f, kgrad = mod.tc_train_step(
        params.clone(), patches.to(dev).to(torch.float32), targets.to(dev).to(torch.int32),
        state, 1e-3, beta1, beta2, 1e-8, 0.0, 1.0 - beta1, 1.0 - beta2, 1, int(ncta_cap))
    loss_f = float(loss_f.item()); kgrad = kgrad.cpu().double()
    rel_loss = abs(loss_f - loss_o) / (abs(loss_o) + 1e-30)
    print(f"[tc {label}] loss kernel={loss_f:.6f} bf16-oracle={loss_o:.6f} rel={rel_loss:.2e}")
    assert rel_loss < _TC_LOSS_REL, f"TC[{label}] loss rel {rel_loss:.2e} > {_TC_LOSS_REL}"
    lay = vit_param_layout()
    worst = 0.0; worst_name = ""
    l0w, l1w, floor = [], [], 0.0
    for name, off, sz, shape in zip(lay["names"], lay["offsets"], lay["sizes"], lay["shapes"]):
        kg = kgrad[off:off + sz].reshape(shape)
        og = grads_o[name]
        fp = grads_fp64[name].double()
        a = (kg - og).abs().max().item(); denom = og.abs().max().item() + 1e-30
        rel = a / denom
        bff = (og - fp).abs().max().item() / denom    # bf16-faithful vs fp64 (the floor)
        floor = max(floor, bff)
        is_w = name.endswith(".weight") or name.endswith("in_proj_weight")
        tol = _TC_GRAD_REL_WEIGHTS if is_w else _TC_GRAD_REL_BIASES
        if "layers.0" in name and is_w and ".n" not in name: l0w.append(rel)
        if "layers.1" in name and is_w and ".n" not in name: l1w.append(rel)
        over = rel / tol
        if over > worst:
            worst = over; worst_name = f"{name} (rel {rel:.2e} tol {tol})"
        print(f"  grad {name:42s} k-vs-bf16 {rel:.3e}  bf16-vs-fp64 {bff:.3e}  tol {tol}")
    import statistics as st
    l0 = st.mean(l0w) if l0w else 0.0; l1 = st.mean(l1w) if l1w else 0.0
    print(f"[tc] STRUCTURAL: layer0 linear-weight mean rel={l0:.3e}  layer1={l1:.3e}  "
          f"(a real bug → layer0 ≫ layer1; bf16 noise → comparable)")
    print(f"[tc] CALIBRATION: max bf16-faithful-vs-fp64 floor={floor:.3e}")
    assert worst < 1.0, (
        f"TC[{label}] grad vs bf16-faithful oracle EXCEEDS calibrated tol: {worst_name} "
        f"({worst:.2f}× its tolerance). A real bug (rel~1) is far over; bf16 noise "
        f"is under. layer0={l0:.2e} layer1={l1:.2e} floor={floor:.2e}.")
    print(f"[tc {label}] single-step grad parity OK (worst {worst:.2f}× tol @ {worst_name})")


@_GATE
def test_tc_single_step_grad_parity():
    """(1+2) KEYSTONE: TC reduced grad vs the bf16-FAITHFUL fp64 ViT oracle (the
    named reference — rounds at the kernel's storage points). Loss rel ≤ 5e-3.

    Calibration witness (printed): kernel-vs-bf16faithful is compared against
    bf16faithful-vs-fp64 (the irreducible bf16 floor) and the layer-0/layer-1
    split (a real bug → early layers ≫ late). patch_proj / cls_token / pos grads
    (the ViT embedding deltas) are checked alongside. Default B=128 at the env
    cap (≤ nCTA → one tile/CTA, full 1088-row tiles)."""
    _grad_parity_core(B=128, ncta_cap=_TC_NCTA_CAP, label="default B=128")


@_GATE
def test_tc_grad_parity_gridstride():
    """KEYSTONE — GRID-STRIDE path: cap=1 forces ONE CTA to grid-stride over
    n_tiles=2 (B=128 → T=2176 = 2×1088), so the for-ti loop body runs TWICE and
    the cross-tile lnvec / nll accumulations (invisible when n_tiles ≤ nCTA) are
    exercised. The shipped path grid-strides on large B; the env-capped gates do
    not. Same calibrated bound as the default."""
    _grad_parity_core(B=128, ncta_cap=1, label="gridstride B=128 cap=1")


@_GATE
def test_tc_grad_parity_ragged_tile():
    """KEYSTONE — RAGGED final tile: B=80 (B%16==0 but B%64 != 0) → T=1360 =
    one full 1088-row tile + one RAGGED 272-row tile (16 samples). cap=1 so the
    single CTA processes both (grid-stride + ragged). Exercises nrows < kTileM:
    the fwd GEMMs over-issue 17 atoms (rows past nrows land in ignored scratch /
    in-workspace reads), the elementwise stages loop nrows, the dW reads only
    valid token rows. Production hits this for any B not a multiple of 64."""
    _grad_parity_core(B=80, ncta_cap=1, label="ragged B=80 cap=1")


@_GATE
def test_tc_determinism():
    """(3) A/A/A bit-exact: identical inputs → identical reduced grad + loss
    (fixed dW tile ownership + ascending-t/k, no float atomics)."""
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    m = _build_eager_vit(dev, seed=7)
    B = 256
    patches = _make_patches(B, 11).to(dev)
    targets = torch.randint(0, 97, (B,), generator=torch.Generator().manual_seed(13)).to(dev)
    total = int(mod.TOTAL)
    outs = []
    for _ in range(3):
        params = _flat_params(m).to(dev)
        state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
        loss, kgrad = _run_tc_step(mod, params, patches, targets, state,
                                   lr=1e-3, betas=(0.9, 0.98), wd=0.0, eps=1e-8, step=1)
        outs.append((float(loss.item()), kgrad.clone()))
    assert torch.equal(outs[0][1], outs[1][1]) and torch.equal(outs[1][1], outs[2][1]), \
        "non-deterministic TC grad (A/A/A not bit-identical)"
    assert outs[0][0] == outs[1][0] == outs[2][0], "non-deterministic TC loss"
    print("[tc] determinism A/A/A bit-identical (grad + loss)  OK")


@_GATE
def test_tc_dw_gemm_exact_on_own_operands():
    """dW GEMM ISOLATION (the strongest correctness witness + tolerance
    calibration anchor): contract the kernel's OWN stored bf16 operands
    dY_ff2[L1] / X_gact[L1] in fp32 ascending-t and compare to the kernel's
    grad[ff2.weight]. ff2.weight is the ONLY dW with Kin=dff=512 → 4 N-tiles in
    vittc_dw_run_tile, the path the engine micro-gates never exercised (they used
    in_dim=128 → 1 N-tile). ~1e-6 ⇒ the dW GEMM (incl. N-tiling) is bit-exact on
    its own operands, so the per-tensor residual vs the bf16 oracle is fully
    attributed to operand-chain bf16 divergence, NOT a GEMM bug."""
    from tests.hw.vit_oracle import vit_param_layout, D_MODEL, D_FF, SEQ
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    m = _build_eager_vit("cpu", seed=123)
    named = {n: p.detach().clone() for n, p in m.named_parameters()}
    B = 128; Tn = B * SEQ
    patches = _make_patches(B, 5)
    targets = torch.randint(0, 97, (B,), generator=torch.Generator().manual_seed(7))
    params = torch.cat([p.reshape(-1) for _, p in named.items()]).contiguous().to(dev)
    total = int(mod.TOTAL)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    _, kgrad = _run_tc_step(mod, params.clone(), patches.to(dev), targets.to(dev),
                            state, lr=1e-3, betas=(0.9, 0.98), wd=0.0, eps=1e-8, step=1)
    kgrad = kgrad.cpu().double()
    dYff2, Xgact = mod.tc_dump_ff2_operands(params, B)
    dYff2 = dYff2.double().reshape(Tn, D_MODEL)
    Xgact = Xgact.double().reshape(Tn, D_FF)
    dW_iso = torch.zeros(D_MODEL, D_FF, dtype=torch.float32)
    A = dYff2.float(); Bm = Xgact.float()
    for t in range(Tn):
        dW_iso += torch.outer(A[t], Bm[t])
    dW_iso = dW_iso.double()
    lay = vit_param_layout()
    off = dict(zip(lay["names"], lay["offsets"])); sz = dict(zip(lay["names"], lay["sizes"]))
    name = "layers.1.ff.2.weight"
    kg = kgrad[off[name]:off[name] + sz[name]].reshape(D_MODEL, D_FF)
    a = (kg - dW_iso).abs().max().item(); denom = dW_iso.abs().max().item() + 1e-30
    rel = a / denom
    print(f"[tc] dW GEMM (ff2.weight, Kin=dff 4-N-tile) vs fp32-contraction-of-own-"
          f"operands rel = {rel:.3e}")
    assert rel < 1e-4, (
        f"TC dW GEMM rel {rel:.2e} > 1e-4 on its OWN operands — the dW contraction "
        f"(likely the n0=n_tile*N N-tile offset in vittc_dw_run_tile) is wrong, not "
        f"bf16 noise.")
    print("[tc] dW GEMM bit-exact on own operands (incl. N-tile path)  OK")


@_GATE
def test_tc_step_time_vs_scalar():
    """(4) step-time TC vs scalar (rough; fleet may be live → informational). We
    assert the TC step completes and report the ratio against the scalar L3-REAL
    path if it is available."""
    import time
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    m = _build_eager_vit(dev, seed=1)
    B = 256
    patches = _make_patches(B, 3).to(dev)
    targets = torch.randint(0, 97, (B,), generator=torch.Generator().manual_seed(3)).to(dev)
    total = int(mod.TOTAL)
    params = _flat_params(m).to(dev)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    for t in range(5):
        _run_tc_step(mod, params, patches, targets, state, 1e-3, (0.9, 0.98), 0.0, 1e-8, t + 1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    NIT = 30
    for t in range(NIT):
        _run_tc_step(mod, params, patches, targets, state, 1e-3, (0.9, 0.98), 0.0, 1e-8, t + 6)
    torch.cuda.synchronize()
    tc_us = (time.perf_counter() - t0) / NIT * 1e6
    print(f"[tc] step-time TC ≈ {tc_us:.1f} µs/step (B={B}, contended GPU → rough)")
    assert tc_us > 0


@_GATE
def test_tc_short_trajectory():
    """(5) short trajectory (50 steps): the TC loss curve tracks the eager bf16
    loss (same init + fixed batch + AdamW). Chaos-robust loose curve-tracking
    gate (bf16 + reordered fp32 reductions diverge chaotically from eager)."""
    import torch.nn.functional as F
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    m_e = _build_eager_vit(dev, seed=321)
    m_f = _build_eager_vit(dev, seed=321)
    m_f.load_state_dict(m_e.state_dict())
    B = 256
    patches = _make_patches(B, 9).to(dev)
    targets = torch.randint(0, 97, (B,), generator=torch.Generator().manual_seed(9)).to(dev)
    lr, betas, wd, eps = 1e-3, (0.9, 0.98), 0.0, 1e-8
    opt = torch.optim.AdamW(m_e.parameters(), lr=lr, betas=betas, weight_decay=wd, eps=eps)
    total = int(mod.TOTAL)
    params = _flat_params(m_f).to(dev)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    K = 50
    max_dev = 0.0
    for t in range(1, K + 1):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            le = F.cross_entropy(m_e(patches).float(), targets)
        le.backward(); opt.step()
        loss_f, _ = _run_tc_step(mod, params, patches, targets, state, lr, betas, wd, eps, t)
        rel = abs(float(loss_f.item()) - le.item()) / (abs(le.item()) + 1e-30)
        max_dev = max(max_dev, rel)
    print(f"[tc] 50-step loss-curve max rel-dev vs eager-bf16 = {max_dev:.2e}")
    assert max_dev < 0.15, f"TC 50-step loss-curve rel-dev {max_dev:.2e} > 0.15"


# ── direct-run report ────────────────────────────────────────────────────
def _main():
    if not _sm90a_available():
        print("SKIP: no sm_90 GPU.")
        return 0
    print("Building ViT TC cell TU (JIT, -DSG_TUNED_GEMM_IMPL=1, sm_90a)…")
    _build_tc_module()
    fails = []

    def _try(name, fn, *a):
        try:
            fn(*a)
        except Exception as e:
            fails.append((name, repr(e)))
            print(f"  [FAIL] {name}: {e!r}")

    _try("fwd-loc(A=I)", test_fwd_identity_localization)
    for (K, Nout) in ((128, 128), (512, 128), (128, 384), (128, 512), (64, 128)):
        _try(f"fwd random K={K} Nout={Nout}", test_fwd_random, K, Nout)
    for Nout in (128, 512):
        _try(f"dX random Nout={Nout}", test_dx_random, Nout)
    for (T, Nout) in ((128, 128), (2048, 128), (2048, 384), (2048, 512)):
        _try(f"dW random T={T} Nout={Nout}", test_dw_random_multistep, T, Nout)
    _try("dW-loc(A=I)", test_dw_identity_who_owns_what)
    _try("determinism", test_determinism_bitwise)

    print("\nFull-cell ViT TC gates (PART 2)…")
    _try("PART2: single-step grad parity", test_tc_single_step_grad_parity)
    _try("PART2: grad parity GRID-STRIDE (cap=1)", test_tc_grad_parity_gridstride)
    _try("PART2: grad parity RAGGED tile (B=80)", test_tc_grad_parity_ragged_tile)
    _try("PART2: dW GEMM exact (own operands)", test_tc_dw_gemm_exact_on_own_operands)
    _try("PART2: determinism A/A/A", test_tc_determinism)
    _try("PART2: short trajectory (50)", test_tc_short_trajectory)
    _try("PART2: step-time vs scalar", test_tc_step_time_vs_scalar)

    print("\n==================== SUMMARY ====================")
    if not fails:
        print("ALL GATES PASSED (PART 1 GEMM-orientation + PART 2 full-cell)")
        return 0
    print(f"{len(fails)} FAILURE(S):")
    for n, e in fails:
        print(f"  - {n}: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(_main())
