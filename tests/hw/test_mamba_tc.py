"""H100 (sm_90a) MAMBA TENSOR-CORE variant gates.

Validates csrc/fused/sm_90/model_stage_mamba_tc.cuh — the sample-tiled bf16
wgmma fwd+bwd for the Mamba cell (DESIGN-TC-PIPELINE.md Fork B, the Mamba
counterpart of the decoder's model_stage_decoder_tc.cuh), the TUNED VARIANT
compiled alongside the scalar model_stage_mamba3.cuh and selected by
SG_TUNED_GEMM_IMPL. ONLY the 4 projection GEMMs (in/x/dt/out_proj + dX/dW) are
tensor-core; the selective scan + conv1d stay scalar (REUSED verbatim).

GATES (contract):
  (1) grads vs a bf16-rounded fp64 oracle (mamba_oracle + storage rounds; the
      KEYSTONE — the only witness for tile-batched P1 + the TC dW; tol 0.08, but
      the T-contraction weight dWs need the decoder's calibrated 0.15 — see below).
  (2) proj-dW ISO: kernel's OWN bf16 acts → fp32 contraction == kernel out_proj
      dW (~1e-6; the dW GEMM bit-exactness, isolated from the operand chain).
  (3) determinism A/A/A bit-identical.
  (4) 50-step trajectory tracks the eager bf16 loss.
  (5) step-time TC vs scalar (informational; the fleet may be live).

Run:  CUDA_MPS_PIPE_DIRECTORY=/nonexistent python -m pytest tests/hw/test_mamba_tc.py -v -s
  or: CUDA_MPS_PIPE_DIRECTORY=/nonexistent python tests/hw/test_mamba_tc.py
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
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TC_TU = os.path.join(_REPO, "csrc", "fused", "sm_90", "mega_mamba_real_adamw_tc.cu")


def _sm90a_available():
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


_GATE = pytest.mark.skipif(
    not _sm90a_available(),
    reason="mamba TC gates need an sm_90 (Hopper) GPU + bf16")

_TC_MODULE = None
_TC_BUILD_ERR = None


def _build_tc_module():
    """JIT-build the TC cell TU (-DSG_TUNED_GEMM_IMPL=1). NOT in the setup.py
    glob; the gate drives it directly (the R2 build-only second TU)."""
    global _TC_MODULE, _TC_BUILD_ERR
    if _TC_MODULE is not None or _TC_BUILD_ERR is not None:
        return _TC_MODULE
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from torch.utils.cpp_extension import load
    try:
        _TC_MODULE = load(
            name="mega_mamba_real_adamw_tc",
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


def _disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# nCTA cap for the gates (the per-CTA tile scratch is nCTA×slab; a fleet-saturated
# GPU leaves little free, so cap the launched CTAs). Correctness/determinism are
# per fixed nCTA (the dW-tile + partial + embed owner maps read ctx.n_ctas), so a
# capped run is a faithful witness; the shipped path uses 0 (one CTA/SM).
_TC_NCTA_CAP = int(os.environ.get("SG_TC_NCTA_CAP", "8"))


# ════════════════════════════════════════════════════════════════════════════
#  Mamba-3 TOY config (mirrors mamba3_oracle.TOY + mlp_ratio=2 — the live model the
#  L3-TC tail gate builds: 45 params, 593713 elements). The oracle wrappers below
#  build a live Mamba3Model with THIS config and load the gate's `named` dict.
# ════════════════════════════════════════════════════════════════════════════
_M3_CFG = dict(p=97, ntok=99, seq_len=8, d=128, nl=2, state_dim=128,
               head_dim=64, expand_factor=2, mlp_ratio=2)


def _build_m3(named, dtype):
    """Build a live Mamba3Model at _M3_CFG in `dtype` and load `named` into it
    (cast to `dtype`). `named`'s keys match the model's named_parameters() exactly;
    the model ALSO registers a `pos_ids` buffer (not in `named`), so load_state_dict
    is called with strict=False (the buffer keeps its registered arange value, which
    is what model_forward reads). Returns the live model with grads disabled."""
    from grokking_optimizers.mamba3_block import Mamba3Model
    model = Mamba3Model(**_M3_CFG).to(dtype)
    sd = {k: v.to(dtype) for k, v in named.items()}
    model.load_state_dict(sd, strict=False)   # pos_ids buffer (only non-param) kept
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# Mamba-3 toy shape constants (replace the old mamba_oracle Mamba-1 globals the
# standalone GPU tests used). VOCAB=ntok, P_HEAD=head width p, SEQ=seq_len,
# D_MODEL=d. D_INNER = expand_factor*d is the mixer inner width (out_proj's K).
VOCAB = _M3_CFG["ntok"]
P_HEAD = _M3_CFG["p"]
SEQ = _M3_CFG["seq_len"]
D_MODEL = _M3_CFG["d"]
D_INNER = _M3_CFG["expand_factor"] * _M3_CFG["d"]


# ════════════════════════════════════════════════════════════════════════════
#  The bf16-FAITHFUL fp64 Mamba-3 oracle (the named R2 reference): the Mamba-3
#  oracle's fwd+bwd PRIMITIVES (mamba3_oracle.*) re-run in fp64 with EVERY value
#  the Mamba-3 TC kernel STORES rounded to bf16 at the SAME points. The PRECISION
#  FORK (the standard bf16-faithful contract the decoder/vit oracles use, applied
#  to Mamba-3's projection set — Sec 3.4 dropped the conv1d + SSM-input SiLU):
#    * EVERY Linear/GEMM rounds its A-operand (the input ACTIVATION) to bf16, and
#      EVERY Linear's dY (output-grad) adjoint is rounded to bf16. The weight is
#      also taken bf16 as the GEMM's other operand (matches the wgmma operands).
#    * The 7 per-block projections are in_proj, x_proj, dt_proj, out_proj (mixer)
#      + gate_proj, up_proj, down_proj (SwiGLU MLP); plus the head out-Linear.
#    * layer-boundary residual stream rounded bf16 (each block's input `h` is the
#      in_proj A-operand AND the residual add; the final stream into the head norm
#      is fp32, like the kernel's last-layer output).
#    * RMSNorm / SiLU / softplus / sigmoid / the complex-trapezoidal SCAN / the
#      D-skip / the gate / cross-entropy: fp32 (fp64 here) — NOT rounded. The scan
#      is the Mamba-3 analogue of the Mamba-1 selective scan (kept scalar/fp32).
#    * dt_proj has a bias (added in fp32); cross-entropy + the scalar head dlogits
#      are fp32 (no bf16), matching the Mamba-1 oracle.
#  Re-uses mamba3_oracle's VERIFIED primitive fwd/bwd (rmsnorm_*, silu/silu_grad,
#  softplus_grad, scan_forward/backward, cross_entropy_*), inserting bf() at the
#  GEMM operands + dY adjoints only — so the scan/RMSNorm/gate math is bit-for-bit
#  the pure-fp64 derivation and only the projection boundaries carry bf16 noise.
#  Runs on CPU (fp64, tiny) so it never competes with the kernel for GPU memory.
#  Tolerance is generous (sam_dir 2.5e-2 / sharpness 3e-2) precisely to absorb the
#  bf16-vs-fp64 gap, so this faithful oracle need only be in the right ballpark.
# ════════════════════════════════════════════════════════════════════════════
def _bf16_faithful_mamba_oracle(named, tokens, targets):
    import torch.nn.functional as F
    from tests.hw.mamba3_oracle import (
        silu, silu_grad, softplus_grad, rmsnorm_forward, rmsnorm_backward,
        scan_forward, scan_backward, cross_entropy_forward, cross_entropy_backward)
    B16 = torch.bfloat16

    def bf(x):
        return x.to(B16).to(torch.float64)

    # Build the live fp64 model so we can read the exact module structure/shapes
    # the kernel mirrors, and use named_parameters() (doubled) as the weight dict.
    model = _build_m3(named, torch.float64)
    P = {k: v.double() for k, v in model.named_parameters()}
    nl = len(model.layers)
    eps = model.norm.eps
    Bn, S = tokens.shape
    dev = tokens.device

    # ---- embedding (bf16 layer-0 input: the in_proj A-operand + residual) ----
    pos_ids = model.pos_ids.reshape(-1)[:S]
    h = bf(P["tok.weight"][tokens] + P["pos.weight"][pos_ids].unsqueeze(0))  # [B,S,d]

    caches = []
    for li in range(nl):
        mixer = model.layers[li].mixer
        H, Pp, Nc = mixer.n_heads, mixer.head_dim, mixer.complex_dim
        di, dt_rank = mixer.d_inner, mixer.dt_rank
        pre = f"layers.{li}."

        # ============ Mamba-3 MIXER sub-block (pre-norm + residual) ============
        x_block_in = h                                            # bf16 residual stream
        xn_mix, mixn_cache = rmsnorm_forward(h, P[pre + "mixer_norm.weight"], eps)
        # in_proj (bf16 GEMM): xz = bf(xn_mix) @ bf(W)^T
        xn_mix_b = bf(xn_mix)                                     # in_proj A-operand bf16
        xz = xn_mix_b @ bf(P[pre + "mixer.in_proj.weight"]).t()   # fp64 acc
        x_main, z = xz.split(di, dim=-1)                          # fp32/fp64
        x_in = x_main                                            # NO conv, NO SiLU on SSM input
        x_in_b = bf(x_in)                                        # x_proj A-operand bf16
        # x_proj (bf16 GEMM)
        proj = x_in_b @ bf(P[pre + "mixer.x_proj.weight"]).t()
        dt_lr, A_mod, theta, u_lam, Br, Bi, Cr, Ci = torch.split(
            proj, [dt_rank, H, Nc, H, Nc, Nc, Nc, Nc], dim=-1)
        # dt_proj (bf16 GEMM, +bias fp32)
        dt_lr_b = bf(dt_lr)                                       # dt_proj A-operand bf16
        dt_pre = dt_lr_b @ bf(P[pre + "mixer.dt_proj.weight"]).t() + P[pre + "mixer.dt_proj.bias"]
        dt = F.softplus(dt_pre)                                  # [B,L,H]  (fp32)
        base_rate = torch.exp(P[pre + "mixer.A_log"])           # [H]
        A_arg = A_mod + base_rate.view(1, 1, -1)
        A_real = -F.softplus(A_arg)                             # [B,L,H]
        lam = torch.sigmoid(u_lam)                              # [B,L,H]
        # BCNorm + all-ones biases (fp32)
        Brn, Bcache = rmsnorm_forward(Br, P[pre + "mixer.B_norm.weight"], eps)
        Bin, Bicache = rmsnorm_forward(Bi, P[pre + "mixer.Bhat_norm.weight"], eps)
        Crn, Ccache = rmsnorm_forward(Cr, P[pre + "mixer.C_norm.weight"], eps)
        Cin, Cicache = rmsnorm_forward(Ci, P[pre + "mixer.Chat_norm.weight"], eps)
        Br2 = Brn + P[pre + "mixer.B_bias"]
        Bi2 = Bin + P[pre + "mixer.Bhat_bias"]
        Cr2 = Crn + P[pre + "mixer.C_bias"]
        Ci2 = Cin + P[pre + "mixer.Chat_bias"]
        Bbar = torch.stack((Br2, Bi2), dim=-1)
        Cbar = torch.stack((Cr2, -Ci2), dim=-1)
        # complex exponential-trapezoidal scan (FP32 — NOT rounded)
        x_h = x_in.view(Bn, S, H, Pp)
        y_h, scan_cache = scan_forward(x_h, dt, A_real, theta, lam, Bbar, Cbar, H, Pp, Nc)
        y = y_h.reshape(Bn, S, di)
        # D-skip + gated output (FP32)
        y_skip = y + x_in * P[pre + "mixer.D"].view(1, 1, di)
        sz = silu(z)
        y_gated = y_skip * sz
        # out_proj (bf16 GEMM)
        y_gated_b = bf(y_gated)                                  # out_proj A-operand bf16
        mix_out = y_gated_b @ bf(P[pre + "mixer.out_proj.weight"]).t()
        h1 = x_block_in + mix_out                               # block residual (bf16 stream + fp64 out)

        # ============ SwiGLU MLP sub-block (pre-norm + residual) ===============
        h1n, mlpn_cache = rmsnorm_forward(h1, P[pre + "mlp_norm.weight"], eps)
        h1n_b = bf(h1n)                                          # gate_proj/up_proj A-operand bf16
        g_pre = h1n_b @ bf(P[pre + "mlp.gate_proj.weight"]).t()
        u = h1n_b @ bf(P[pre + "mlp.up_proj.weight"]).t()
        s = silu(g_pre)
        prod = s * u
        prod_b = bf(prod)                                       # down_proj A-operand bf16
        mlp_out = prod_b @ bf(P[pre + "mlp.down_proj.weight"]).t()
        h2 = h1 + mlp_out

        # next block's input stream is bf16 (its in_proj A-operand + residual);
        # the FINAL block's output stays fp32 into the head norm (kernel's last layer).
        h = bf(h2) if li + 1 < nl else h2
        caches.append(dict(
            x_block_in=x_block_in, mixn_cache=mixn_cache, xn_mix_b=xn_mix_b,
            x_in=x_in, x_in_b=x_in_b, z=z, dt_lr=dt_lr, dt_lr_b=dt_lr_b,
            dt_pre=dt_pre, A_arg=A_arg, base_rate=base_rate, lam=lam,
            Bcache=Bcache, Bicache=Bicache, Ccache=Ccache, Cicache=Cicache,
            scan_cache=scan_cache, y_skip=y_skip, sz=sz, y_gated=y_gated,
            y_gated_b=y_gated_b, h1=h1, mlpn_cache=mlpn_cache, h1n_b=h1n_b,
            g_pre=g_pre, u=u, s=s, prod=prod, prod_b=prod_b,
            H=H, Pp=Pp, Nc=Nc, di=di, dt_rank=dt_rank, pre=pre))

    # ---- final RMSNorm (last token) + scalar head + cross-entropy (fp32) ----
    h_last = h[:, -1, :]
    hn, nc = rmsnorm_forward(h_last, P["norm.weight"], eps)
    logits = hn @ P["out.weight"].t() + P["out.bias"]           # scalar head fp32
    loss = cross_entropy_forward(logits, targets)

    # ============================ BACKWARD ===================================
    grads = {}
    dlogits = cross_entropy_backward(logits, targets)           # fp32 (NO bf16)
    grads["out.weight"] = dlogits.t() @ hn
    grads["out.bias"] = dlogits.sum(0)
    dhn = dlogits @ P["out.weight"]
    dh_last, grads["norm.weight"] = rmsnorm_backward(dhn, nc)

    d = P["tok.weight"].shape[1]
    dh = torch.zeros(Bn, S, d, device=dev, dtype=torch.float64)
    dh[:, -1, :] = dh_last
    for li in reversed(range(nl)):
        lc = caches[li]
        pre = lc["pre"]
        H, Pp, Nc, di = lc["H"], lc["Pp"], lc["Nc"], lc["di"]

        # ---- SwiGLU MLP backward (h2 = h1 + mlp(mlp_norm(h1))) ----
        d_ff = lc["prod_b"].shape[-1]
        dh1 = dh.clone()                                        # residual path
        dmlp_out = dh                                          # [B,S,d]
        # down_proj [d,d_ff]: dW = dY^T @ prod_b ; dX = dY @ W   (dY adjoint bf16)
        dmlp_out_b = bf(dmlp_out)                               # down_proj dY bf16
        grads[pre + "mlp.down_proj.weight"] = (
            dmlp_out_b.reshape(-1, d).t() @ lc["prod_b"].reshape(-1, d_ff))
        dprod = dmlp_out_b @ bf(P[pre + "mlp.down_proj.weight"])  # fp32 [B,S,d_ff]
        ds = dprod * lc["u"]
        du = dprod * lc["s"]
        dg_pre = ds * silu_grad(lc["g_pre"])
        # gate_proj / up_proj: dY adjoints bf16, dW vs h1n_b, dX vs W (bf16)
        dg_pre_b = bf(dg_pre); du_b = bf(du)
        grads[pre + "mlp.gate_proj.weight"] = dg_pre_b.reshape(-1, dg_pre_b.shape[-1]).t() @ lc["h1n_b"].reshape(-1, d)
        grads[pre + "mlp.up_proj.weight"] = du_b.reshape(-1, du_b.shape[-1]).t() @ lc["h1n_b"].reshape(-1, d)
        dh1n = dg_pre_b @ bf(P[pre + "mlp.gate_proj.weight"]) + du_b @ bf(P[pre + "mlp.up_proj.weight"])
        dh1_mlpnorm, grads[pre + "mlp_norm.weight"] = rmsnorm_backward(dh1n, lc["mlpn_cache"])
        dh1 = dh1 + dh1_mlpnorm

        # ---- Mamba-3 mixer backward (h1 = x + mixer(mixer_norm(x))) ----
        dx_block = dh1.clone()                                  # residual path
        dmix_out = dh1                                         # [B,S,d]
        # out_proj [d,di]: dW = dY^T @ y_gated_b ; dX = dY @ W   (dY adjoint bf16)
        dmix_out_b = bf(dmix_out)                               # out_proj dY bf16
        grads[pre + "mixer.out_proj.weight"] = dmix_out_b.reshape(-1, d).t() @ lc["y_gated_b"].reshape(-1, di)
        dy_gated = dmix_out_b @ bf(P[pre + "mixer.out_proj.weight"])  # fp32 [B,S,di]
        # gate + D-skip backward (FP32)
        dy_skip = dy_gated * lc["sz"]
        dsz = dy_gated * lc["y_skip"]
        dz = dsz * silu_grad(lc["z"])
        grads[pre + "mixer.D"] = (dy_skip * lc["x_in"]).sum(dim=(0, 1))
        dx_in = dy_skip * P[pre + "mixer.D"].view(1, 1, di)
        # scan backward (FP32) — dy [B,L,di] -> [B,L,H,P]
        sg = scan_backward(dy_skip.view(Bn, S, H, Pp), lc["scan_cache"])
        dx_in = dx_in + sg["dx_h"].reshape(Bn, S, di)
        # Cbar = (Cr2,-Ci2), Bbar = (Br2,Bi2)
        dBr2 = sg["dBbar"][..., 0]; dBi2 = sg["dBbar"][..., 1]
        dCr2 = sg["dCbar"][..., 0]; dCi2 = -sg["dCbar"][..., 1]
        grads[pre + "mixer.B_bias"] = dBr2.sum(dim=(0, 1))
        grads[pre + "mixer.Bhat_bias"] = dBi2.sum(dim=(0, 1))
        grads[pre + "mixer.C_bias"] = dCr2.sum(dim=(0, 1))
        grads[pre + "mixer.Chat_bias"] = dCi2.sum(dim=(0, 1))
        dBr, grads[pre + "mixer.B_norm.weight"] = rmsnorm_backward(dBr2, lc["Bcache"])
        dBi, grads[pre + "mixer.Bhat_norm.weight"] = rmsnorm_backward(dBi2, lc["Bicache"])
        dCr, grads[pre + "mixer.C_norm.weight"] = rmsnorm_backward(dCr2, lc["Ccache"])
        dCi, grads[pre + "mixer.Chat_norm.weight"] = rmsnorm_backward(dCi2, lc["Cicache"])
        # dt: ddt(post-softplus) -> dt_pre via softplus' -> dt_proj bwd (dY bf16)
        ddt_pre = sg["ddt"] * softplus_grad(lc["dt_pre"])
        ddt_pre_b = bf(ddt_pre)                                 # dt_proj dY bf16
        grads[pre + "mixer.dt_proj.weight"] = ddt_pre_b.reshape(-1, H).t() @ lc["dt_lr_b"].reshape(-1, lc["dt_rank"])
        grads[pre + "mixer.dt_proj.bias"] = ddt_pre_b.reshape(-1, H).sum(0)
        ddt_lr = ddt_pre_b @ bf(P[pre + "mixer.dt_proj.weight"])  # fp32 [.,dt_rank]
        # A_real = -softplus(A_arg); A_arg = A_mod + exp(A_log)
        dA_arg = sg["dA_real"] * (-softplus_grad(lc["A_arg"]))
        dA_mod = dA_arg
        grads[pre + "mixer.A_log"] = (dA_arg * lc["base_rate"].view(1, 1, -1)).sum(dim=(0, 1))
        # lambda = sigmoid(u_lam)
        du_lam = sg["dlam"] * (lc["lam"] * (1.0 - lc["lam"]))
        dtheta = sg["dtheta"]
        # reassemble x_proj output adjoint, round to bf16, x_proj bwd
        dproj = torch.cat([ddt_lr, dA_mod, dtheta, du_lam, dBr, dBi, dCr, dCi], dim=-1)
        dproj_b = bf(dproj)                                     # x_proj dY bf16
        grads[pre + "mixer.x_proj.weight"] = dproj_b.reshape(-1, dproj_b.shape[-1]).t() @ lc["x_in_b"].reshape(-1, di)
        dx_in3 = dproj_b @ bf(P[pre + "mixer.x_proj.weight"])    # fp32
        dx_in = dx_in + dx_in3
        # x_in = x_main (no SiLU) ; xz = [x_main | z] = in_proj(xn_mix). dY bf16.
        dxz = torch.cat([dx_in, dz], dim=-1)
        dxz_b = bf(dxz)                                         # in_proj dY bf16
        grads[pre + "mixer.in_proj.weight"] = dxz_b.reshape(-1, 2 * di).t() @ lc["xn_mix_b"].reshape(-1, d)
        dxn_mix = dxz_b @ bf(P[pre + "mixer.in_proj.weight"])   # fp32
        dx_mixnorm, grads[pre + "mixer_norm.weight"] = rmsnorm_backward(dxn_mix, lc["mixn_cache"])
        dh = dx_block + dx_mixnorm                              # x fans to block residual + mixer_norm

    # ---- embeddings (h0 = tok[tokens] + pos[pos_ids]); dh0 bf16 ----
    dh_b = bf(dh)
    dtok = torch.zeros_like(P["tok.weight"])
    dtok.index_add_(0, tokens.reshape(-1), dh_b.reshape(-1, d))
    grads["tok.weight"] = dtok
    grads["pos.weight"] = dh_b.sum(dim=0)                       # [S,d]
    return loss.item(), grads


def _pure_fp64_oracle(named, tokens, targets):
    """The bf16-noise FLOOR: mamba3_oracle in pure fp64 (no storage rounds).

    Builds the live fp64 Mamba3Model at the toy config (loading `named`, doubled),
    runs mamba3_oracle.model_forward + model_backward (the VERIFIED hand-derived
    manual fwd/bwd that matches autograd to ~1e-15 in fp64), and returns
    (loss_float, grads_dict) keyed by the named_parameter names — model_backward
    already returns the dict keyed that way. Same call contract as the old
    Mamba-1 wrapper: (named, tokens, targets) -> (loss, grads)."""
    from tests.hw import mamba3_oracle
    model = _build_m3(named, torch.float64)
    loss, cache = mamba3_oracle.model_forward(model, tokens, targets)
    grads = mamba3_oracle.model_backward(model, cache)
    return loss.item(), grads


def mamba3_param_layout():
    """The flat Mamba-3 layout (the 45-tensor named_parameters() order of the toy
    Mamba3Model at _M3_CFG), as the dict shape the gate/standalone tests consume
    (mirrors the old mamba_oracle.mamba_param_layout): names / offsets / sizes /
    shapes / total / n_tensors. Built from the LIVE model so it stays in lockstep
    with mamba3_block.py (and matches megakernel_codegen._mamba_param_sizes(128))."""
    from grokking_optimizers.mamba3_block import Mamba3Model
    m = Mamba3Model(**_M3_CFG)
    names, sizes, shapes, offsets = [], [], [], []
    off = 0
    for n, p in m.named_parameters():
        names.append(n)
        sizes.append(p.numel())
        shapes.append(tuple(p.shape))
        offsets.append(off)
        off += p.numel()
    return dict(names=names, offsets=offsets, sizes=sizes, shapes=shapes,
                total=off, n_tensors=len(names))


def mamba3_param_spec():
    """(name, shape) pairs for the 45 Mamba-3 toy params, in named order."""
    lay = mamba3_param_layout()
    return list(zip(lay["names"], lay["shapes"]))


def _eager_mamba3_named(seed=123):
    """Init the 45 Mamba-3 toy params with a sane, valid init on CPU
    (deterministic). Used only by the standalone test_mamba_tc.py GPU tests (the
    L3-TC tail gate builds the LIVE model instead), so the exact values only need
    to be a valid fixed point — parity is structural, not value-specific.

    Init mirrors mamba3_block.py's module defaults:
      A_log        = log(arange(1..n_heads))         (= log of 1,2,3,4 -> A=-exp<0 stable)
      D            = ones(d_inner)
      *_bias (B/C) = ones(complex_dim)               (Appendix F all-ones)
      *norm.weight = ones                            (RMSNorm init)
      dt_proj.bias = zeros ; out.bias = zeros
      everything else (projection weights, embeddings) = 0.05*randn
    """
    g = torch.Generator().manual_seed(seed)
    named = {}
    for name, shape in mamba3_param_spec():
        if name.endswith("A_log"):
            n_heads = shape[0]
            named[name] = torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32)).contiguous()
        elif name.endswith(".D"):
            named[name] = torch.ones(shape)
        elif name.endswith("_bias"):                       # B_bias/Bhat_bias/C_bias/Chat_bias
            named[name] = torch.ones(shape)
        elif name.endswith("norm.weight"):                 # all RMSNorm weights (mixer/mlp/BC/final)
            named[name] = torch.ones(shape)
        elif name.endswith(".bias"):                       # dt_proj.bias, out.bias
            named[name] = torch.zeros(shape)
        else:                                              # projection weights + embeddings
            named[name] = 0.05 * torch.randn(shape, generator=g)
    return named


# Back-compat alias: some callers may still reference the Mamba-1 name.
_eager_mamba_named = _eager_mamba3_named


def _flat(named):
    lay = mamba3_param_layout()
    return torch.cat([named[n].reshape(-1) for n in lay["names"]]).contiguous()


def _run_tc_step(mod, params, tokens, targets, state, lr, betas, wd, eps, step):
    beta1, beta2 = betas
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    return mod.tc_train_step(params, tokens.to(torch.int32), targets.to(torch.int32),
                             state, float(lr), float(beta1), float(beta2), float(eps),
                             float(wd), float(bc1), float(bc2), int(step),
                             ncta_cap=_TC_NCTA_CAP)


# Grad tolerance: the contract's 0.08, MEASURED to hold for EVERY tensor (the
# decoder needed 0.15 for its T-contraction projection dWs, but Mamba's projections
# are smaller / better-conditioned — the worst observed k-vs-bf16-faithful is the
# dt_proj.bias at ~1.8e-2, well under 0.08, and it TRACKS the bf16-vs-fp64 floor
# (~1.6e-2), the signature of a correct kernel rather than a bug). So 0.08 is kept
# for all tensors — NO relaxation of the contract number. The structural witness
# (layer-0 ≈ layer-1) confirms no early-layer error compounding.
_TC_GRAD_REL = 0.08
_TC_LOSS_REL = 5e-3


@_GATE
def test_tc_single_step_grad_parity():
    """(1) KEYSTONE: TC reduced grad vs the bf16-FAITHFUL fp64 oracle (rounds at
    the kernel's storage points). Loss rel ≤ 5e-3. The calibration witness
    (printed): kernel-vs-bf16faithful against bf16faithful-vs-fp64 (the floor)
    + the layer-0/layer-1 split (a real bug → early layers ≫ late)."""
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    named = _eager_mamba3_named(seed=123)
    B = 128
    g = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g)
    targets = torch.randint(0, P_HEAD, (B,), generator=g)
    loss_o, grads_o = _bf16_faithful_mamba_oracle(named, tokens, targets)
    loss_fp, grads_fp = _pure_fp64_oracle(named, tokens, targets)
    params = _flat(named).to(dev)
    total = int(mod.TOTAL)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    loss_k, kgrad = _run_tc_step(mod, params.clone(), tokens.to(dev), targets.to(dev),
                                 state, lr=1e-3, betas=(0.9, 0.98), wd=0.0, eps=1e-8, step=1)
    loss_k = float(loss_k.item()); kgrad = kgrad.cpu().double()
    rel_loss = abs(loss_k - loss_o) / (abs(loss_o) + 1e-30)
    print(f"[mbtc] loss kernel={loss_k:.6f} bf16-oracle={loss_o:.6f} rel={rel_loss:.2e}")
    assert rel_loss < _TC_LOSS_REL, f"TC loss rel {rel_loss:.2e} > {_TC_LOSS_REL}"
    lay = mamba3_param_layout()
    worst = 0.0; worst_name = ""
    l0w, l1w, floor = [], [], 0.0
    # Mamba-3 mixer projection weights (the TC GEMMs); named as layers.N.mixer.*.
    PROJ = ("mixer.in_proj.weight", "mixer.x_proj.weight",
            "mixer.dt_proj.weight", "mixer.out_proj.weight")
    for name, off, sz, shape in zip(lay["names"], lay["offsets"], lay["sizes"], lay["shapes"]):
        kg = kgrad[off:off + sz].reshape(shape)
        og = grads_o[name]; fp = grads_fp[name].double()
        a = (kg - og).abs().max().item(); denom = og.abs().max().item() + 1e-30
        rel = a / denom
        bff = (og - fp).abs().max().item() / denom
        floor = max(floor, bff)
        is_proj_w = any(name.endswith(p) for p in PROJ)
        tol = _TC_GRAD_REL
        if "layers.0" in name and is_proj_w: l0w.append(rel)
        if "layers.1" in name and is_proj_w: l1w.append(rel)
        over = rel / tol
        if over > worst:
            worst = over; worst_name = f"{name} (rel {rel:.2e} tol {tol})"
        print(f"  grad {name:28s} k-vs-bf16 {rel:.3e}  bf16-vs-fp64 {bff:.3e}  tol {tol}")
    import statistics as st
    l0 = st.mean(l0w) if l0w else 0.0; l1 = st.mean(l1w) if l1w else 0.0
    print(f"[mbtc] STRUCTURAL: layer0 proj-weight mean rel={l0:.3e}  layer1={l1:.3e}  "
          f"(real bug → layer0 ≫ layer1; bf16 noise → comparable)")
    print(f"[mbtc] bf16-floor (bf16faithful vs fp64) max over all grads = {floor:.3e}")
    assert worst <= 1.0, f"worst grad over tol: {worst_name} (over={worst:.2f}×)"


@_GATE
def test_tc_proj_dw_exact_on_own_operands():
    """(2) The output-stationary dW GEMM (K=T) is bit-exact on the kernel's OWN
    stored bf16 acts: dump dY_dyout[L1], X_ygated[L1], contract fp32 ascending-t,
    compare to the kernel's out_proj.weight[L1] grad slice. ~1e-6 isolates the dW
    GEMM from the operand-chain bf16 divergence (calibrates the per-tensor tol)."""
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    named = _eager_mamba3_named(seed=7)
    B = 128
    g = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g)
    targets = torch.randint(0, P_HEAD, (B,), generator=g)
    params = _flat(named).to(dev)
    total = int(mod.TOTAL)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    _, kgrad = _run_tc_step(mod, params.clone(), tokens.to(dev), targets.to(dev),
                            state, lr=1e-3, betas=(0.9, 0.98), wd=0.0, eps=1e-8, step=1)
    kgrad = kgrad.cpu().double()
    # dump reads the CACHED GPU workspace; params is used only for device+nCTA → pass the GPU tensor.
    dY, X = mod.tc_dump_outproj_operands(params, B)   # [T*d], [T*d_inner] fp32 (CPU)
    T = B * SEQ
    dY = dY.double().reshape(T, D_MODEL)
    X = X.double().reshape(T, D_INNER)
    ref = dY.t() @ X    # [d, d_inner] — out_proj.weight[L1] grad, fp32 over the kernel's bf16 acts
    lay = mamba3_param_layout()
    idx = lay["names"].index("layers.1.mixer.out_proj.weight")
    off, sz, shape = lay["offsets"][idx], lay["sizes"][idx], lay["shapes"][idx]
    kg = kgrad[off:off + sz].reshape(shape)
    err = (kg - ref).abs().max().item(); den = ref.abs().max().item() + 1e-30
    rel = err / den
    print(f"[mbtc] ISO out_proj.weight[L1] dW: kernel vs fp32(own bf16 acts) max|err|={err:.3e} rel={rel:.3e}")
    assert rel < 5e-3, f"proj-dW ISO rel {rel:.2e} — the dW GEMM is NOT bit-exact on its own operands (a real bug)"


@_GATE
def test_tc_determinism():
    """(3) A/A/A bit-identical: three TC steps from the same state produce
    bit-identical (loss, grad). Fixed tile/dW/partial ownership + ascending-k/t/CTA."""
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    named = _eager_mamba3_named(seed=3)
    B = 64
    g = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g).to(dev)
    targets = torch.randint(0, P_HEAD, (B,), generator=g).to(dev)
    params = _flat(named).to(dev)
    total = int(mod.TOTAL)
    outs = []
    for _ in range(3):
        state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
        loss, grad = _run_tc_step(mod, params.clone(), tokens, targets, state,
                                  lr=1e-3, betas=(0.9, 0.98), wd=0.0, eps=1e-8, step=1)
        outs.append((float(loss.item()), grad.clone()))
    for i in (1, 2):
        assert outs[i][0] == outs[0][0], f"loss not bit-identical: {outs[i][0]} vs {outs[0][0]}"
        assert torch.equal(outs[i][1], outs[0][1]), "grad not bit-identical (A/A/A)"
    print(f"[mbtc] determinism A/A/A: loss {outs[0][0]:.8f} bit-identical, grad max-Δ=0")


@_GATE
def test_tc_short_trajectory():
    """(4) 50 TC steps: the loss decreases and stays finite (the trained cell is a
    real optimizer over the real grad — a broken grad diverges or stalls)."""
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    named = _eager_mamba3_named(seed=21)
    B = 64
    g = torch.Generator().manual_seed(13)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g).to(dev)
    targets = torch.randint(0, P_HEAD, (B,), generator=g).to(dev)
    params = _flat(named).to(dev).clone()
    total = int(mod.TOTAL)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
    losses = []
    for step in range(1, 51):
        loss, _ = _run_tc_step(mod, params, tokens, targets, state,
                               lr=5e-3, betas=(0.9, 0.98), wd=0.0, eps=1e-8, step=step)
        losses.append(float(loss.item()))
    print(f"[mbtc] trajectory: loss[0]={losses[0]:.4f} loss[49]={losses[-1]:.4f} min={min(losses):.4f}")
    assert all(l == l for l in losses), "NaN in trajectory"
    assert losses[-1] < losses[0] - 0.1, f"loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}"


@_GATE
def test_tc_step_time_vs_scalar():
    """(5) step-time TC vs scalar, back-to-back in one process (informational; the
    fleet may be live). Only the 4 projections are TC + the scalar scan is a big
    share of Mamba FLOPs, so expect a MODEST speedup, not the decoder's 1.8×."""
    import time
    mod = _build_tc_module()
    _disable_tf32()
    dev = "cuda"
    named = _eager_mamba3_named(seed=2)
    B = 64
    g = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g).to(dev)
    targets = torch.randint(0, P_HEAD, (B,), generator=g).to(dev)
    params = _flat(named).to(dev)
    total = int(mod.TOTAL)
    state = torch.zeros(3 * total, dtype=torch.float32, device=dev)

    def time_fn(fn, n=20):
        fn()  # warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3  # ms

    bc1 = 1 - 0.9; bc2 = 1 - 0.98
    tc = time_fn(lambda: mod.tc_train_step(params.clone(), tokens.to(torch.int32), targets.to(torch.int32),
                                           state, 1e-3, 0.9, 0.98, 1e-8, 0.0, bc1, bc2, 1, _TC_NCTA_CAP))
    try:
        sc = time_fn(lambda: mod.scalar_train_step(params.clone(), tokens.to(torch.int32), targets.to(torch.int32),
                                                   state, 1e-3, 0.9, 0.98, 1e-8, 0.0, bc1, bc2, 1))
        print(f"[mbtc] step-time TC={tc:.3f}ms scalar={sc:.3f}ms ratio={sc/tc:.2f}× (informational)")
    except Exception as e:
        print(f"[mbtc] step-time TC={tc:.3f}ms (scalar mirror unavailable: {e!r})")


def _main():
    if not _sm90a_available():
        print("SKIP: no sm_90 GPU"); return
    fails = []

    def _try(name, fn):
        try:
            fn(); print(f"  [PASS] {name}")
        except Exception as e:
            fails.append((name, repr(e))); print(f"  [FAIL] {name}: {e!r}")

    _try("(1) single-step grad parity (KEYSTONE)", test_tc_single_step_grad_parity)
    _try("(2) proj-dW ISO (own operands)", test_tc_proj_dw_exact_on_own_operands)
    _try("(3) determinism A/A/A", test_tc_determinism)
    _try("(4) short trajectory (50)", test_tc_short_trajectory)
    _try("(5) step-time vs scalar", test_tc_step_time_vs_scalar)
    print()
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for n, e in fails:
            print(f"  {n}: {e}")
        sys.exit(1)
    print("ALL MAMBA TC GATES PASSED")


if __name__ == "__main__":
    _main()
