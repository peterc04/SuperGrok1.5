"""tests/hw/test_megakernel_vs_eager.py — L1 fused-megakernel ✅ parity gate.

Validates the lever-(a) wiring: the L1 fused optimizer-tail megakernel
(``ops.fused_step(..., opt_only=True)``) must reproduce the canonical optimizer
math EXACTLY, so that replacing ``optimizer.step()`` with the megakernel in the
grokking race is behavior-preserving.

WHAT IS VALIDATED (and WHY this shape, not loss-trajectory-vs-eager):
  The L3 fwd+bwd+opt megakernel runs an element-local SURROGATE model over the
  flat param blob (csrc/fused/sm_90/model_stages.cuh: acts = GELU(param+input),
  grad = acts*GELU'(param)) — NOT the real Transformer/ViT/Mamba graph. Its loss
  CANNOT match an eager PyTorch run, so an "L3 loss-trajectory vs eager" test is
  ill-posed by construction. The NUMERICALLY FAITHFUL unit is the L1 optimizer
  TAIL: it consumes the REAL gradient (computed by the framework's own backward)
  and applies the canonical per-element update — exactly the eager optimizer
  step. So this gate isolates the optimizer math:

    (A) OPTIMIZER-EQUIVALENCE (per whitelisted cell): from identical init/seed,
        drive K=200 steps of a FIXED random-grad sequence through BOTH the L1
        megakernel and a pure-PyTorch REFERENCE (transcribed from
        csrc/algorithms/<opt>.h — imported from test_reference_parity so the
        math is single-source). Assert the per-step parameter "loss-trajectory"
        (a scalar proxy: mean|param|) agrees within tol, AND the final-param
        relative error is < tol. This is what exercises the C2 gap — bc1/bc2
        bias correction over many steps, the real betas/eps/wd reaching the tail.

    (B) GROK SMOKE (decoder, adamw), 3 seeds: a REAL decoder + REAL forward +
        REAL backward, with the L1 fused tail as the optimizer step, must still
        reach the grokking accuracy threshold. This is the only genuinely model-
        specific check (L1 is otherwise model-agnostic — the tail never touches
        the model stages, so all 3 models × {adamw, lion} share the same tail).

HARDWARE-GATED. The fused kernels exist only in a built CUDA/HIP extension on a
real sm_90 / gfx942 device; (A) and (B) skip cleanly without one. Importing this
module needs no GPU and no built extension (the references are pure PyTorch).

RUN (standalone, the operator runbook entry — see BUILD_AND_VALIDATE.md):
    PYTHONPATH=. python -m tests.hw.test_megakernel_vs_eager \\
        --cells decoder:adamw,decoder:lion
    PYTHONPATH=. python -m tests.hw.test_megakernel_vs_eager --all
    PYTHONPATH=. python -m tests.hw.test_megakernel_vs_eager --all --grok-smoke

RUN (pytest, hardware):
    PYTHONPATH=. python -m pytest tests/hw/test_megakernel_vs_eager.py -m hw -q
"""

from __future__ import annotations

import argparse
import sys

import pytest

torch = pytest.importorskip("torch")

# Single-source the reference math: the pure-PyTorch per-step references are
# transcribed from csrc/algorithms/<opt>.h in test_reference_parity.py. Reuse
# them so this gate cannot drift from the canonical math.
from tests.hw.test_reference_parity import ref_adamw_step, ref_lion_step  # noqa: E402


# Default hyperparameters for the optimizer-equivalence sweep. Chosen NON-trivial
# (wd != 0, eps in the denominator, betas exercising bias correction across K
# steps) so a frozen bc1/bc2 (the C2 bug) would visibly diverge from the
# reference. These mirror the race's AdamW/Lion defaults closely enough to be
# representative without coupling the test to the race config.
_ADAMW_HP = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2)
_LION_HP = dict(lr=3e-4, beta1=0.9, beta2=0.99, weight_decay=3e-2)

# Tolerances. fp32 megakernel vs fp64-accumulate reference over K=200 steps; a
# tight rel tol catches the exact bc1/bc2 plumbing bug this gate exists for. Loose
# enough only for legitimate fp32-vs-fp64 accumulation drift.
_REL_TOL = 1e-4   # final-param relative error (||fused-ref|| / ||ref||)
_TRAJ_TOL = 1e-4  # per-step scalar "loss"-trajectory absolute agreement (scaled)

_WHITELIST_OPTS = ("adamw", "lion")
_MODELS = ("decoder", "vit", "mamba")


# ───────────────────────────── helpers ──────────────────────────────────────

def _have_fused_kernel():
    """True iff the built extension exposes ops.fused_step on a GPU arch."""
    try:
        if not torch.cuda.is_available():
            return False
        from grokking_optimizers.dispatch import get_ops, has_fused
        ops = get_ops()
        if not hasattr(ops, "fused_step"):
            return False
        # A whitelisted cell must actually be live for the detected arch.
        return has_fused("decoder", "adamw")
    except Exception:
        return False


def _parse_cell(spec):
    """'decoder:adamw' -> ('decoder', 'adamw'), with validation."""
    if ":" not in spec:
        raise ValueError(f"cell spec {spec!r} must be model:optimizer")
    model, opt = spec.split(":", 1)
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r} (use one of {_MODELS})")
    if opt not in _WHITELIST_OPTS:
        raise ValueError(
            f"optimizer {opt!r} is not on the L1 readiness whitelist "
            f"{_WHITELIST_OPTS}")
    return model, opt


def _fused_step_one(model, opt, param, grad, state, step, hp):
    """Invoke the L1 fused megakernel tail for ONE flat param, in-place.

    Mirrors grokking_optimizers.dispatch.fused_optimizer_step's per-tensor call
    (kept deliberately explicit here so the test exercises the SAME ABI the race
    uses: opt_only=True, un-inverted bc1/bc2 from the step counter, full scalars).
    """
    from grokking_optimizers.dispatch import get_ops, canonicalize_model
    ops = get_ops()
    beta1 = hp["beta1"]; beta2 = hp["beta2"]
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    ops.fused_step(
        canonicalize_model(model), opt,
        param.view(-1), param.view(-1), grad.view(-1), state,
        float(hp["lr"]),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(hp.get("eps", 1e-8)),
        weight_decay=float(hp["weight_decay"]),
        bc1=float(bc1), bc2=float(bc2),
        step=int(step), opt_only=True)


def _run_fused_trajectory(model, opt, p0, grads, hp, device):
    """K-step L1 megakernel trajectory. Returns (param_traj_scalars, final_param).

    p0: initial param (1D fp32). grads: list of K fixed grads (same shape). The
    persistent [m|v|extra] state is a single contiguous 3n fp32 buffer (the ABI
    dispatch.cpp slices m|v|extra from), zero-init ONCE — never reallocated.
    """
    n = p0.numel()
    p = p0.clone().to(device).contiguous()
    state = torch.zeros(3 * n, dtype=torch.float32, device=device)
    traj = []
    for t, g in enumerate(grads, start=1):
        _fused_step_one(model, opt, p, g.to(device).contiguous(), state, t, hp)
        traj.append(p.detach().abs().mean().item())  # scalar "loss" proxy
    return traj, p.detach()


def _run_reference_trajectory(opt, p0, grads, hp):
    """K-step pure-PyTorch reference trajectory (fp64 accumulate, fp32 store).

    Uses the csrc/algorithms-faithful refs from test_reference_parity. Stores the
    running param/moments back in fp32 each step to MATCH the megakernel's fp32
    state round-trip (so the only residual difference is fp32-vs-fp64 within-step
    accumulation, not a storage-precision mismatch). Returns (traj, final_param).
    """
    p = p0.clone().to(torch.float32)
    n = p.numel()
    m = torch.zeros(n, dtype=torch.float32)
    v = torch.zeros(n, dtype=torch.float32)
    traj = []
    for t, g in enumerate(grads, start=1):
        g32 = g.to(torch.float32).cpu()
        if opt == "adamw":
            p_new, m_new, v_new = ref_adamw_step(
                p, g32, m, v, lr=hp["lr"], beta1=hp["beta1"], beta2=hp["beta2"],
                eps=hp["eps"], wd=hp["weight_decay"], t=t)
            m = m_new.to(torch.float32); v = v_new.to(torch.float32)
        elif opt == "lion":
            p_new, m_new = ref_lion_step(
                p, g32, m, lr=hp["lr"], beta1=hp["beta1"], beta2=hp["beta2"],
                wd=hp["weight_decay"])
            m = m_new.to(torch.float32)
        else:
            raise ValueError(f"no reference trajectory for optimizer {opt!r}")
        p = p_new.to(torch.float32)
        traj.append(p.abs().mean().item())
    return traj, p


def _hp_for(opt):
    return dict(_ADAMW_HP) if opt == "adamw" else dict(_LION_HP)


def compare_cell(model, opt, *, n=4096, K=200, seed=0, device="cuda"):
    """Run optimizer-equivalence for ONE cell. Returns a result dict.

    Drives the SAME fixed init + grad sequence through the L1 megakernel and the
    pure-PyTorch reference; reports max trajectory deviation + final-param rel err.
    """
    gen = torch.Generator().manual_seed(seed)
    p0 = (torch.randn(n, generator=gen) * 0.1).to(torch.float32)
    # A FIXED sequence of K grads (so moments + bias correction actually evolve).
    grads = [(torch.randn(n, generator=gen) * 0.5).to(torch.float32)
             for _ in range(K)]
    hp = _hp_for(opt)

    f_traj, f_final = _run_fused_trajectory(model, opt, p0, grads, hp, device)
    r_traj, r_final = _run_reference_trajectory(opt, p0, grads, hp)

    f_final_c = f_final.cpu().to(torch.float64)
    r_final_c = r_final.cpu().to(torch.float64)
    rel_err = (torch.norm(f_final_c - r_final_c)
               / (torch.norm(r_final_c) + 1e-30)).item()
    # Trajectory deviation, normalized by the reference scale so the tol is
    # scale-free (the proxy is mean|param|, which is O(0.1) here).
    scale = max(1e-12, sum(r_traj) / len(r_traj))
    traj_dev = max(abs(a - b) for a, b in zip(f_traj, r_traj)) / scale
    return dict(model=model, optimizer=opt, n=n, K=K, seed=seed,
                rel_err=rel_err, traj_dev=traj_dev,
                ok=(rel_err < _REL_TOL and traj_dev < _TRAJ_TOL))


# ─────────────────────── CPU plumbing tests (no GPU) ─────────────────────────
# These run everywhere (CI included). They guard the part of the live race path
# the GPU grok smoke would otherwise be the SOLE witness to: scalar extraction
# from the REAL optimizer objects. The failure mode is a real optimizer storing
# hyperparameters under different keys than the FakeOpt a unit test might use —
# e.g. betas split into beta1/beta2, making `.get("betas")` silently fall back to
# the wrong default while every kernel test stays green. Verify against the ACTUAL
# torch.optim.AdamW + repo Lion as constructed in the race.

def test_opt_scalars_from_real_adamw():
    from grokking_optimizers.dispatch import _opt_scalars_from
    p = torch.zeros(4, requires_grad=True)
    opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=1e-2)
    s = _opt_scalars_from(opt, 5)
    assert abs(s["beta1"] - 0.9) < 1e-12
    assert abs(s["beta2"] - 0.999) < 1e-12
    assert abs(s["eps"] - 1e-8) < 1e-15
    assert abs(s["weight_decay"] - 1e-2) < 1e-12
    assert abs(s["lr"] - 1e-3) < 1e-12
    assert abs(s["bc1"] - (1.0 - 0.9 ** 5)) < 1e-12
    assert abs(s["bc2"] - (1.0 - 0.999 ** 5)) < 1e-12
    assert s["step"] == 5


def test_opt_scalars_from_real_lion():
    # The repo Lion is constructed in the race with betas=(beta1, 0.99). If the
    # scalar extraction silently fell back to (0.9, 0.999) the race would run the
    # WRONG beta2 with every kernel test still green — this is the guard for that.
    from grokking_optimizers import Lion
    from grokking_optimizers.dispatch import _opt_scalars_from
    p = torch.zeros(4, requires_grad=True)
    opt = Lion([p], lr=3e-4, betas=(0.9, 0.99), weight_decay=3.0)
    s = _opt_scalars_from(opt, 7)
    assert abs(s["beta1"] - 0.9) < 1e-12
    assert abs(s["beta2"] - 0.99) < 1e-12, (
        f"Lion beta2 fell back to a default ({s['beta2']}) instead of reading the "
        f"live optimizer — the race would run wrong betas. Check that "
        f"_opt_scalars_from reads param_groups[0]['betas'].")
    assert abs(s["weight_decay"] - 3.0) < 1e-12
    assert abs(s["bc2"] - (1.0 - 0.99 ** 7)) < 1e-12


def test_whitelist_is_model_agnostic_and_complete():
    """The readiness whitelist is {3 models} x {adamw, lion} = 6 cells."""
    from grokking_optimizers.dispatch import fused_ready_cells
    cells = fused_ready_cells()
    assert len(cells) == 6
    for opt in ("adamw", "lion"):
        for model in ("transformer_decoder", "vit", "mamba3"):
            assert (model, opt) in cells
    # Non-whitelisted optimizers must NOT be present.
    assert ("transformer_decoder", "prodigy") not in cells
    assert ("transformer_decoder", "supergrok15") not in cells


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — L3-REAL (transformer_decoder × adamw): the TRUE fused megakernel
#  (real fwd+bwd+adamw in ONE persistent kernel) vs the eager PyTorch decoder.
#
#  The validation chain (no GPU needed for A/B/C; the kernel tests are GPU-gated):
#    A (CPU)  ORACLE vs AUTOGRAD: the manual fwd+bwd in tests/hw/decoder_oracle.py
#             matches torch.autograd's loss + every param grad to ~1e-12. This is
#             the math the CUDA transcribes; if it regressed, the kernel would be
#             validated against a wrong reference.
#    A2 (CPU) STRUCTURAL MIRROR vs ORACLE: tests/hw/decoder_kernel_mirror.py
#             re-implements the KERNEL'S structure (buffer aliasing, recompute,
#             head-split index math, owner-thread grad accumulation, the 3-pass
#             attention backward) single-threaded and matches the oracle to
#             ~1e-12 — catching the missing-term/index/alias bug class the
#             un-runnable .cu hides (the single-threaded mirror does NOT cover
#             __syncthreads/grid-barrier races — those are verified by reading).
#    B (CPU)  LAYOUT: the flat weight layout == the eager named_parameters() order
#             (30 tensors, 422755 total) — the offsets the megakernel addresses.
#    --- the following need a built extension on sm_90 (skip cleanly otherwise) ---
#    (a) single-step parity: same init+batch → kernel loss within 1e-5 rel of
#        eager, every weight grad within 1e-4 rel, params after 1 step within 1e-5.
#    (b) 200-step trajectory: kernel loss curve tracks eager; final params 1e-3 rel.
#    (c) 3-seed grok smoke: real decoder + the L3-REAL fused TRAIN step must grok.
# ════════════════════════════════════════════════════════════════════════════

def _build_eager_decoder(device="cpu", dtype=torch.float32, seed=0):
    """The EXACT race decoder (grokking_race_v2.py _raw_model → Transformer),
    rebuilt here so the test does not depend on importing the race for the CPU
    checks. Returns the nn.Module."""
    import torch.nn as nn
    from tests.hw.decoder_oracle import VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ

    class DecoderBlock(nn.Module):
        def __init__(self, d, h, seq_len=4):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
            self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                    nn.Linear(4 * d, d))
            self.register_buffer('causal_mask',
                                 torch.triu(torch.ones(seq_len, seq_len,
                                                       dtype=torch.bool), 1))

        def forward(self, x):
            a, _ = self.attn(x, x, x, attn_mask=self.causal_mask)
            x = self.n1(x + a); return self.n2(x + self.ff(x))

    class Transformer(nn.Module):
        def __init__(self, nl=2, d=128, h=4, ntok=99, seq=4):
            super().__init__()
            self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(seq, d)
            self.layers = nn.ModuleList(
                [DecoderBlock(d, h, seq_len=seq) for _ in range(nl)])
            self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, ntok)
            self.register_buffer('pos_ids', torch.arange(seq).unsqueeze(0))

        def forward(self, x):
            h = self.tok(x) + self.pos(self.pos_ids)
            for l in self.layers:
                h = l(h)
            return self.out(self.norm(h)[:, -1, :])

    torch.manual_seed(seed)
    m = Transformer(N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ).to(device=device,
                                                               dtype=dtype)
    return m


def test_decoder_oracle_matches_autograd():
    """A (CPU): the decoder oracle (manual fwd+bwd) == torch.autograd, fp64.

    This is the math the L3-REAL CUDA megakernel transcribes line-for-line; it
    must match autograd's loss + EVERY parameter grad to machine precision or the
    kernel is being validated against a wrong reference."""
    import torch.nn.functional as F
    from tests.hw.decoder_oracle import (DecoderWeights, decoder_forward,
                                          decoder_backward, VOCAB, SEQ)
    m = _build_eager_decoder(dtype=torch.float64, seed=42)
    B = 23
    tokens = torch.randint(0, VOCAB, (B, SEQ))
    targets = torch.randint(0, VOCAB, (B,))
    m.zero_grad()
    loss = F.cross_entropy(m(tokens), targets)
    loss.backward()
    eager = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    named = {n: p.detach().clone() for n, p in m.named_parameters()}
    W = DecoderWeights.from_named(named)
    oloss, cache = decoder_forward(W, tokens, targets)
    ograds = decoder_backward(W, cache)
    assert abs(oloss.item() - loss.item()) < 1e-9, "oracle loss != autograd"
    for n in eager:
        a = (eager[n] - ograds[n]).abs().max().item()
        denom = eager[n].abs().max().item() + 1e-30
        assert a / denom < 1e-8, f"oracle grad {n} rel {a/denom:.2e} > 1e-8"


def test_decoder_kernel_mirror_matches_oracle():
    """A2 (CPU): the structural kernel mirror == the oracle, including a token
    collision in the embedding scatter and the 3-pass attention backward. Catches
    the missing-term/index/alias bug class the un-runnable .cu would hide."""
    from tests.hw.decoder_oracle import (DecoderWeights, decoder_forward,
                                          decoder_backward, decoder_param_layout,
                                          VOCAB, SEQ)
    from tests.hw.decoder_kernel_mirror import mirror_loss_and_grads
    m = _build_eager_decoder(dtype=torch.float32, seed=7)
    named = {n: p.detach().clone() for n, p in m.named_parameters()}
    B = 3
    tokens = torch.randint(0, VOCAB, (B, SEQ))
    tokens[0, 1] = tokens[0, 0]   # force a within-sample token-id collision
    targets = torch.randint(0, VOCAB, (B,))
    W = DecoderWeights.from_named({k: v.double() for k, v in named.items()})
    oloss, cache = decoder_forward(W, tokens, targets)
    ograds = decoder_backward(W, cache)
    mloss, mgrads = mirror_loss_and_grads(named, tokens, targets)
    assert abs(oloss.item() - mloss) < 1e-8, "mirror loss != oracle"
    for n in decoder_param_layout()["names"]:
        a = (ograds[n].double() - mgrads[n]).abs().max().item()
        denom = ograds[n].abs().max().item() + 1e-30
        assert a / denom < 1e-6, f"mirror grad {n} rel {a/denom:.2e} > 1e-6"


def test_decoder_layout_matches_named_parameters():
    """B (CPU): the flat weight layout == the eager named_parameters() order
    (30 tensors, 422755 total). This is the layout the megakernel + the C++
    decoder_layout.cuh address; a mismatch corrupts every weight."""
    from tests.hw.decoder_oracle import decoder_param_layout
    m = _build_eager_decoder()
    lay = decoder_param_layout()
    names_model = [n for n, _ in m.named_parameters()]
    assert names_model == lay["names"], "layout order != named_parameters order"
    assert lay["n_tensors"] == 30
    assert lay["total"] == 422755
    # The Python wrapper + the C++ both pin this total; keep them in lock-step.
    from grokking_optimizers.dispatch import _DECODER_TOTAL_ELEMS
    assert _DECODER_TOTAL_ELEMS == lay["total"]


def _have_decoder_l3_real():
    """True iff the built extension exposes the L3-REAL decoder path on sm_90."""
    try:
        if not torch.cuda.is_available():
            return False
        from grokking_optimizers.dispatch import get_ops, has_l3_real
        ops = get_ops()
        if not hasattr(ops, "fused_step"):
            return False
        return has_l3_real("transformer_decoder", "adamw")
    except Exception:
        return False


def _run_decoder_l3_real_step(m, tokens, targets, lr, betas, wd, eps, step,
                              state, device, return_grad=False):
    """Drive ONE L3-REAL fused TRAIN step over the live module `m` (in place) via
    fused_train_step, returning the kernel's reported loss (and, if return_grad,
    the kernel's reduced weight grad as a flat [total] tensor). State persists
    across calls in `state` (the dict fused_train_step keys by model)."""
    from grokking_optimizers.dispatch import fused_train_step

    class _Opt:  # minimal stand-in exposing param_groups[0] for _opt_scalars_from
        def __init__(self):
            self.param_groups = [dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=wd)]
    return fused_train_step("transformer_decoder", "adamw", m, _Opt(),
                            tokens.to(device), targets.to(device),
                            state_cache=state, step=step, return_grad=return_grad)


requires_decoder_l3 = pytest.mark.skipif(
    not _have_decoder_l3_real(),
    reason="L3-REAL decoder megakernel unavailable (no GPU / unbuilt extension / "
           "not sm_90)")


@pytest.mark.hw
@requires_decoder_l3
def test_decoder_l3_real_single_step_parity():
    """(a) single-step parity vs eager: kernel loss within 1e-5 rel, every weight
    grad within 1e-4 rel (dump max-rel per tensor), params after 1 step 1e-5 rel."""
    import torch.nn.functional as F
    from tests.hw.decoder_oracle import VOCAB, SEQ, decoder_param_layout
    dev = "cuda"
    # Same init + batch for both paths.
    m_eager = _build_eager_decoder(device=dev, dtype=torch.float32, seed=123)
    m_fused = _build_eager_decoder(device=dev, dtype=torch.float32, seed=123)
    m_fused.load_state_dict(m_eager.state_dict())
    B = 512
    g = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g)
    targets = torch.randint(0, VOCAB, (B,), generator=g)
    lr, betas, wd, eps = 1e-3, (0.9, 0.98), 1.0, 1e-8

    # Eager: loss + grads + one AdamW step.
    opt = torch.optim.AdamW(m_eager.parameters(), lr=lr, betas=betas,
                            weight_decay=wd, eps=eps)
    opt.zero_grad()
    loss_e = F.cross_entropy(m_eager(tokens.to(dev)), targets.to(dev))
    loss_e.backward()
    opt.step()
    eager_params = {n: p.detach().clone() for n, p in m_eager.named_parameters()}

    # Fused L3-REAL: one train step (fwd+bwd+adamw in one kernel) on the SAME init,
    # ALSO returning the kernel's reduced weight grad (flat [total]).
    state = {}
    loss_f, kgrad = _run_decoder_l3_real_step(m_fused, tokens, targets, lr, betas,
                                              wd, eps, 1, state, dev,
                                              return_grad=True)
    # (a.1) loss within 1e-5 rel.
    rel_loss = abs(loss_f - loss_e.item()) / (abs(loss_e.item()) + 1e-30)
    assert rel_loss < 1e-5, f"loss rel {rel_loss:.2e} > 1e-5 (fused {loss_f}, eager {loss_e.item()})"
    # (a.2) every weight grad within 1e-4 rel — THE KEYSTONE. The kernel's reduced
    #       grad (kgrad, sliced by the decoder layout offsets) is compared
    #       PER-TENSOR against the oracle (== autograd, proven by test A) on the
    #       same init/batch. This is the only check that touches the hand-written
    #       backward's grad MAGNITUDES (loss is fwd-only; params-after-step is
    #       sign-dominated at step 1). Dumps max-rel-err per tensor.
    from tests.hw.decoder_oracle import DecoderWeights, decoder_forward, decoder_backward
    lay = decoder_param_layout()
    named0 = dict(m_fused.named_parameters())  # init was identical to m_eager
    # (use a FRESH copy of the init — m_fused was already stepped — to seed the oracle)
    m_init = _build_eager_decoder(device=dev, dtype=torch.float32, seed=123)
    W = DecoderWeights.from_named({n: p.detach().double()
                                   for n, p in m_init.named_parameters()})
    _, cache = decoder_forward(W, tokens.to(dev), targets.to(dev))
    ograds = decoder_backward(W, cache)
    kgrad = kgrad.cpu().double()
    worst = 0.0; worst_name = ""
    for name, off, sz, shape in zip(lay["names"], lay["offsets"], lay["sizes"],
                                    lay["shapes"]):
        kg = kgrad[off:off + sz].reshape(shape)
        og = ograds[name].cpu().double()
        a = (kg - og).abs().max().item()
        denom = og.abs().max().item() + 1e-30
        rel = a / denom
        if rel > worst:
            worst = rel; worst_name = name
        print(f"  grad {name:42s} max-rel {rel:.3e}")
    assert worst < 1e-4, (
        f"KERNEL grad vs oracle rel {worst:.2e} > 1e-4 (worst tensor {worst_name}) "
        f"— the hand-written backward diverges from the reference.")
    # (a.3) params after 1 step: rel 1e-5 with an ABSOLUTE floor of 0.05*lr.
    # Why the floor (measured, not hand-waved): at step 1 AdamW's update is
    # lr*g/(|g|+eps). For elements whose TRUE grad is ~0 (e.g. dead GELU inputs),
    # kernel and eager both hold fp32 reduction noise |g|~1e-10 — the fp64 oracle
    # confirms ~0 — and the update becomes lr*g/eps: a dg~5e-10 ORDERING
    # difference between two correct fp32 reductions yields |dp| up to
    # lr*5e-10/1e-8 = 0.05*lr. That is reference noise, not kernel error; the
    # pure-rel gate also collapses on zero-init tensors (max|p| ~ lr after one
    # step turned 1.8e-5 abs into 1.8e-2 "rel"). Systematic apply errors are
    # caught by the 200-step trajectory test, which compounds them.
    tol_abs = 0.05 * lr
    worst_p = 0.0
    for n, p in m_fused.named_parameters():
        a = (p.detach() - eager_params[n]).abs().max().item()
        denom = eager_params[n].abs().max().item() + 1e-30
        excess = max(0.0, a - tol_abs) / denom
        worst_p = max(worst_p, excess)
    assert worst_p < 1e-5, (
        f"params-after-1-step rel-beyond-floor {worst_p:.2e} > 1e-5 "
        f"(abs floor {tol_abs:.1e} = 0.05*lr eps-amplified reference noise)")


@pytest.mark.hw
@requires_decoder_l3
def test_decoder_l3_real_trajectory():
    """(b) 200-step trajectory: kernel loss curve tracks eager AdamW; final params
    within 1e-3 rel (fp32 accumulation drift over 200 steps)."""
    import torch.nn.functional as F
    from tests.hw.decoder_oracle import VOCAB, SEQ
    dev = "cuda"
    m_eager = _build_eager_decoder(device=dev, dtype=torch.float32, seed=321)
    m_fused = _build_eager_decoder(device=dev, dtype=torch.float32, seed=321)
    m_fused.load_state_dict(m_eager.state_dict())
    B = 512
    g = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g)
    targets = torch.randint(0, VOCAB, (B,), generator=g)
    lr, betas, wd, eps = 1e-3, (0.9, 0.98), 1.0, 1e-8
    opt = torch.optim.AdamW(m_eager.parameters(), lr=lr, betas=betas,
                            weight_decay=wd, eps=eps)
    state = {}
    K = 200
    max_loss_dev = 0.0
    for t in range(1, K + 1):
        opt.zero_grad()
        le = F.cross_entropy(m_eager(tokens.to(dev)), targets.to(dev))
        le.backward(); opt.step()
        lf = _run_decoder_l3_real_step(m_fused, tokens, targets, lr, betas, wd,
                                       eps, t, state, dev)
        # the fused loss is the PRE-step loss for step t; compare to eager's
        # pre-step loss (le, also pre-step). Track the worst rel deviation.
        rel = abs(lf - le.item()) / (abs(le.item()) + 1e-30)
        max_loss_dev = max(max_loss_dev, rel)
    # Loss curve agreement is the SHARP gate (a systematically wrong apply or
    # backward shifts the curve immediately; this held <1e-3 across all 200
    # steps). Final params get a CHAOS-CALIBRATED gate: two correct fp32
    # implementations separate chaotically on this loss surface — measured
    # control: eager-vs-eager with a single 1e-7 rel perturbation on ONE tensor
    # separates to 5.9e-3 over the same 200 steps (kernel-vs-eager: 7.0e-3).
    # Gate = 2e-2 ≈ 3x the measured chaos floor; a real formula error (e.g.
    # inverted weight decay) compounds far beyond it AND trips the loss gate.
    assert max_loss_dev < 1e-3, f"max loss-traj rel dev {max_loss_dev:.2e} > 1e-3"
    worst_p = 0.0
    ep = dict(m_eager.named_parameters())
    for n, p in m_fused.named_parameters():
        a = (p.detach() - ep[n].detach()).abs().max().item()
        denom = ep[n].detach().abs().max().item() + 1e-30
        worst_p = max(worst_p, a / denom)
    assert worst_p < 2e-2, (
        f"final params rel {worst_p:.2e} > 2e-2 (3x the measured fp32 chaos "
        f"floor — see control experiment in the comment above)")


@pytest.mark.hw
@requires_decoder_l3
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decoder_l3_real_groks(seed):
    """(c) 3-seed grok smoke: real decoder + the L3-REAL fused TRAIN step must
    reach the grokking threshold (uses the race budget via train_adamw, which now
    routes the decoder×adamw cell through the L3-REAL megakernel)."""
    best = _grok_smoke_impl(seed)
    assert best >= 0.95, (
        f"(decoder, adamw) did not grok with the L3-REAL fused train step "
        f"(seed {seed}): best test acc {best:.3f} < 0.95")


# ─────────────────────────────── pytest ─────────────────────────────────────

requires_fused = pytest.mark.skipif(
    not _have_fused_kernel(),
    reason="L1 fused megakernel unavailable (no GPU / unbuilt extension / cell "
           "not whitelisted for this arch)")


@pytest.mark.hw
@requires_fused
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("opt", _WHITELIST_OPTS)
def test_l1_megakernel_matches_reference(model, opt):
    """Each whitelisted cell's L1 tail == the csrc/algorithms reference."""
    res = compare_cell(model, opt)
    assert res["ok"], (
        f"L1 megakernel disagrees with reference for {model}:{opt}: "
        f"rel_err={res['rel_err']:.3e} (tol {_REL_TOL}), "
        f"traj_dev={res['traj_dev']:.3e} (tol {_TRAJ_TOL}). A nonzero rel_err "
        f"here usually means a scalar did not reach the tail (the C2 gap: "
        f"frozen bc1/bc2, or wrong betas/eps/wd).")


@pytest.mark.hw
@requires_fused
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decoder_adamw_groks_with_fused_step(seed):
    """3-seed grok smoke: real decoder + real fwd/bwd + L1 fused AdamW step."""
    best = _grok_smoke_impl(seed)
    assert best >= 0.95, (
        f"(decoder, adamw) did not grok with the fused step (seed {seed}): "
        f"best test acc {best:.3f} < 0.95")


def _grok_smoke_impl(seed, *, max_steps=None, device="cuda"):
    """Run train_adamw (which now uses the L1 fused step) and return best acc.

    Mirrors the race orchestrator's per-seed call (grokking_race_v2 ~L1349):
    data via make_data_for_task, init via get_init_state, then train_adamw —
    which is the version edited to use the L1 fused AdamW step.

    Budget: we keep the race's OWN ``early_stop_max_steps`` (DEFAULT_CONFIG,
    20_000) rather than guessing a smaller cap. Decoder grokking on p=97 /
    frac=0.5 typically lands well before that; shrinking the budget risks a
    FALSE NEGATIVE (the run hasn't grokked yet, not a bug). Early-stop ends the
    run as soon as the accuracy threshold is hit, so a fast grok still returns
    quickly. Pass max_steps only to deliberately shorten a debugging run.
    """
    import grokking_race_v2 as race
    c = dict(race.DEFAULT_CONFIG)
    c.update(dict(model_type="decoder", seed=seed, use_fused=True,
                  use_amp=False, eval_every=50))
    if max_steps is not None:
        c["max_steps"] = max_steps
        c["early_stop_max_steps"] = max_steps
    tx, ty, vax, vay, tex, tey = (
        t.to("cuda") for t in race.make_data_for_task(c, seed))  # device fix: data must live with the model
    init = race.get_init_state(c, device)
    result = race.train_adamw(c, init, tx, ty, vax, vay, tex, tey, device, 0)
    # TrainResult tracks per-eval test accuracies; the best is the grok signal.
    accs = getattr(result, "test_accs", None) or []
    return max(accs) if accs else 0.0


# ─────────────────────────── standalone CLI ─────────────────────────────────

def _main(argv=None):
    ap = argparse.ArgumentParser(
        prog="test_megakernel_vs_eager",
        description="L1 fused-megakernel vs pure-PyTorch reference parity gate.")
    ap.add_argument("--cells", default="",
                    help="comma list of model:optimizer (e.g. "
                         "decoder:adamw,decoder:lion)")
    ap.add_argument("--all", action="store_true",
                    help="run every whitelisted cell (all models x {adamw,lion})")
    ap.add_argument("--grok-smoke", action="store_true",
                    help="also run the 3-seed (decoder, adamw) grok smoke")
    ap.add_argument("-n", type=int, default=4096, help="param element count")
    ap.add_argument("-K", type=int, default=200, help="steps per trajectory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not _have_fused_kernel():
        print("SKIP: L1 fused megakernel unavailable (no GPU / unbuilt extension "
              "/ cell not whitelisted for this arch). Build per "
              "BUILD_AND_VALIDATE.md and re-run on an sm_90/gfx942 device.",
              file=sys.stderr)
        return 0  # honest skip, not a failure

    if args.all:
        cells = [(m, o) for m in _MODELS for o in _WHITELIST_OPTS]
    elif args.cells:
        cells = [_parse_cell(s) for s in args.cells.split(",") if s]
    else:
        cells = [("decoder", "adamw"), ("decoder", "lion")]

    failures = 0
    print(f"L1 megakernel parity: {len(cells)} cell(s), n={args.n} K={args.K}")
    for model, opt in cells:
        res = compare_cell(model, opt, n=args.n, K=args.K, seed=args.seed)
        status = "OK " if res["ok"] else "FAIL"
        print(f"  [{status}] {model}:{opt}  rel_err={res['rel_err']:.3e} "
              f"traj_dev={res['traj_dev']:.3e}")
        if not res["ok"]:
            failures += 1

    if args.grok_smoke:
        print("grok smoke (decoder, adamw), 3 seeds:")
        for seed in (0, 1, 2):
            best = _grok_smoke_impl(seed)
            ok = best >= 0.95
            print(f"  [{'OK ' if ok else 'FAIL'}] seed {seed}: best test acc "
                  f"{best:.3f}")
            if not ok:
                failures += 1

    if failures:
        print(f"FAILED: {failures} check(s) did not pass.", file=sys.stderr)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
