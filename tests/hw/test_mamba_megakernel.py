"""tests/hw/test_mamba_megakernel.py — PHASE-2 Mamba L3-REAL megakernel gates.

Structure mirrors tests/hw/test_megakernel_vs_eager.py's decoder gates:

  NO-GPU (run anywhere, the rigor substitute for an un-runnable .cu):
    * test_mamba_oracle_matches_autograd — the manual fwd+bwd oracle == autograd
      (loss + every parameter grad, ~1e-12 fp64). This is the math — INCLUDING the
      selective-scan backward — that the CUDA megakernel transcribes.
    * test_mamba_kernel_mirror_matches_oracle — the single-threaded STRUCTURAL
      mirror of the kernel (per-channel register scan, scan-bwd recompute+reverse,
      NON-causal conv transpose, 3-path dx_main, owner-thread accumulation, a
      forced token collision) == the oracle (~1e-12). Catches the missing-term/
      index/alias bug class the un-runnable .cu hides.
    * test_mamba_layout_matches_named_parameters — the flat weight layout == the
      eager named_parameters() order (28 tensors, 259425 total).

  GPU-gated (sm_90 + built extension; skip cleanly otherwise): the L3-REAL Mamba
  single-step parity + trajectory + grok gates, MIRRORING the decoder GPU gates.
  These are NOT run in this phase (the implementing agent does no GPU work); they
  are written so the operator can run them once the composition/launcher land.

Run the CPU gates:
    PYTHONPATH=. python -m pytest tests/hw/test_mamba_megakernel.py \
        -k "oracle or mirror or layout" -q
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Build the eager MambaModel (the ground-truth module the oracle transcribes).
# ---------------------------------------------------------------------------
def _build_eager_mamba(dtype=torch.float32, seed=42):
    import grokking_race_v2 as g
    torch.manual_seed(seed)
    m = g.MambaModel(p=97, ntok=99, seq_len=8, d=128, nl=2, grad_checkpoint=False)
    return m.to(dtype)


# ===========================================================================
#  A (CPU) ORACLE vs AUTOGRAD
# ===========================================================================
def test_mamba_oracle_matches_autograd():
    """The Mamba oracle (manual fwd+bwd, NO autograd) == torch.autograd, fp64.

    This is the math the L3-REAL CUDA megakernel transcribes line-for-line —
    INCLUDING the selective-scan reverse-time backward. It must match autograd's
    loss + EVERY parameter grad to machine precision or the kernel is being
    validated against a wrong reference."""
    import torch.nn.functional as F
    from tests.hw.mamba_oracle import (MambaWeights, mamba_forward,
                                        mamba_backward, mamba_param_layout,
                                        VOCAB, P_HEAD, SEQ)
    m = _build_eager_mamba(dtype=torch.float64, seed=42)
    B = 11
    tokens = torch.randint(0, VOCAB, (B, SEQ))
    targets = torch.randint(0, P_HEAD, (B,))
    m.zero_grad()
    loss = F.cross_entropy(m(tokens), targets)
    loss.backward()
    eager = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    named = {n: p.detach().clone() for n, p in m.named_parameters()}
    W = MambaWeights.from_named(named)
    oloss, cache = mamba_forward(W, tokens, targets)
    ograds = mamba_backward(W, cache)
    assert abs(oloss.item() - loss.item()) < 1e-9, "oracle loss != autograd"
    for n in mamba_param_layout()["names"]:
        a = (eager[n] - ograds[n]).abs().max().item()
        denom = eager[n].abs().max().item() + 1e-30
        assert a / denom < 1e-8, f"oracle grad {n} rel {a/denom:.2e} > 1e-8"


# ===========================================================================
#  A2 (CPU) STRUCTURAL MIRROR vs ORACLE
# ===========================================================================
def test_mamba_kernel_mirror_matches_oracle():
    """The structural kernel mirror == the oracle, including a within-sample
    token-id collision in the embedding scatter, the per-channel register scan,
    the scan backward (recompute + reverse), the NON-causal conv transpose, and
    the 3-path dx_main accumulation. Catches the missing-term/index/alias bug
    class the un-runnable .cu would hide."""
    from tests.hw.mamba_oracle import (MambaWeights, mamba_forward,
                                        mamba_backward, mamba_param_layout,
                                        VOCAB, P_HEAD, SEQ)
    from tests.hw.mamba_kernel_mirror import mirror_loss_and_grads
    m = _build_eager_mamba(dtype=torch.float32, seed=7)
    named = {n: p.detach().clone() for n, p in m.named_parameters()}
    B = 2
    tokens = torch.randint(0, VOCAB, (B, SEQ))
    tokens[0, 1] = tokens[0, 0]   # force a within-sample token-id collision
    targets = torch.randint(0, P_HEAD, (B,))
    W = MambaWeights.from_named({k: v.double() for k, v in named.items()})
    oloss, cache = mamba_forward(W, tokens, targets)
    ograds = mamba_backward(W, cache)
    mloss, mgrads = mirror_loss_and_grads(named, tokens, targets)
    assert abs(oloss.item() - mloss) < 1e-8, "mirror loss != oracle"
    for n in mamba_param_layout()["names"]:
        a = (ograds[n].double() - mgrads[n]).abs().max().item()
        denom = ograds[n].abs().max().item() + 1e-30
        assert a / denom < 1e-6, f"mirror grad {n} rel {a/denom:.2e} > 1e-6"


# ===========================================================================
#  B (CPU) LAYOUT == named_parameters()
# ===========================================================================
def test_mamba_layout_matches_named_parameters():
    """The flat weight layout == the eager named_parameters() order (28 tensors,
    259425 total). This is the layout the megakernel + the C++ mamba3_layout.cuh
    address; a mismatch corrupts every weight. CRITICAL: A_log/D precede in_proj
    within each layer (module-own params before submodules)."""
    from tests.hw.mamba_oracle import mamba_param_layout
    m = _build_eager_mamba()
    lay = mamba_param_layout()
    names_model = [n for n, _ in m.named_parameters()]
    assert names_model == lay["names"], (
        "layout order != named_parameters order\n"
        f"  model:  {names_model}\n  layout: {lay['names']}")
    assert lay["n_tensors"] == 28
    assert lay["total"] == 259425
    # shapes match too.
    shapes_model = [tuple(p.shape) for _, p in m.named_parameters()]
    assert shapes_model == [tuple(s) for s in lay["shapes"]]


# ===========================================================================
#  GPU-gated L3-REAL gates (sm_90 + built extension). NOT run this phase —
#  written so the operator can run them once the composition/launcher land.
#  Skip cleanly when the extension / device is absent.
# ===========================================================================
def _have_mamba_l3_real():
    """True iff the built extension exposes the L3-REAL Mamba path on sm_90."""
    try:
        if not torch.cuda.is_available():
            return False
        from grokking_optimizers.dispatch import get_ops, has_l3_real
        ops = get_ops()
        if not hasattr(ops, "fused_step"):
            return False
        return has_l3_real("mamba", "adamw")
    except Exception:
        return False


def _run_mamba_l3_real_step(m, tokens, targets, lr, betas, wd, eps, step,
                            state, device, return_grad=False):
    """Drive ONE L3-REAL fused TRAIN step over the live module `m` (in place) via
    fused_train_step, returning the kernel's reported loss (and, if return_grad,
    the kernel's reduced weight grad as a flat [total] tensor). State persists
    across calls in `state` (the dict fused_train_step keys by model). Mirrors the
    decoder's _run_decoder_l3_real_step / the ViT _run_vit_l3_real_step."""
    from grokking_optimizers.dispatch import fused_train_step

    class _Opt:  # minimal stand-in exposing param_groups[0] for _opt_scalars_from
        def __init__(self):
            self.param_groups = [dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=wd)]
    return fused_train_step("mamba", "adamw", m, _Opt(),
                            tokens.to(device), targets.to(device), lr=lr,
                            betas=betas, weight_decay=wd, eps=eps,
                            state_cache=state, step=step, return_grad=return_grad)


requires_mamba_l3 = pytest.mark.skipif(
    not _have_mamba_l3_real(),
    reason="L3-REAL Mamba path unavailable (no GPU / unbuilt extension / not "
           "sm_90 / integrator seam not yet wired — see INTEGRATION-MAMBA.md).")


@pytest.mark.hw
@requires_mamba_l3
def test_mamba_l3_real_single_step_parity():
    """(a) single-step parity vs eager: kernel loss within 1e-5 rel, every weight
    grad within 1e-4 rel (vs the oracle), params after 1 step 1e-5 rel.

    MIRRORS test_decoder_l3_real_single_step_parity. The keystone (a.2) is the
    PER-TENSOR grad comparison of the kernel's reduced weight grad against the
    oracle — the ONLY check that exercises the hand-written selective-scan
    backward's MAGNITUDES (loss is fwd-only; params-after-step is sign-dominated
    at step 1)."""
    import torch.nn.functional as F
    from tests.hw.mamba_oracle import (MambaWeights, mamba_forward,
                                        mamba_backward, mamba_param_layout,
                                        VOCAB, P_HEAD, SEQ)
    dev = "cuda"
    m_eager = _build_eager_mamba(seed=123).to(dev)
    m_fused = _build_eager_mamba(seed=123).to(dev)
    m_fused.load_state_dict(m_eager.state_dict())
    B = 512
    g = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g)
    targets = torch.randint(0, P_HEAD, (B,), generator=g)
    lr, betas, wd, eps = 1e-3, (0.9, 0.98), 1.0, 1e-8

    # Eager: loss + grads + one AdamW step.
    opt = torch.optim.AdamW(m_eager.parameters(), lr=lr, betas=betas,
                            weight_decay=wd, eps=eps)
    opt.zero_grad()
    loss_e = F.cross_entropy(m_eager(tokens.to(dev)), targets.to(dev))
    loss_e.backward()
    opt.step()
    eager_params = {n: p.detach().clone() for n, p in m_eager.named_parameters()}

    # Fused L3-REAL: one train step on the SAME init, ALSO returning the kernel's
    # reduced weight grad (flat [total]).
    state = {}
    loss_f, kgrad = _run_mamba_l3_real_step(m_fused, tokens, targets, lr, betas,
                                            wd, eps, 1, state, dev,
                                            return_grad=True)
    # (a.1) loss within 1e-5 rel.
    rel_loss = abs(loss_f - loss_e.item()) / (abs(loss_e.item()) + 1e-30)
    assert rel_loss < 1e-5, f"loss rel {rel_loss:.2e} > 1e-5 (fused {loss_f}, eager {loss_e.item()})"
    # (a.2) every weight grad within 1e-4 rel vs the oracle (== autograd, test A)
    #       on the same init/batch. THE KEYSTONE — exercises the selective-scan
    #       backward's grad magnitudes. Dumps max-rel-err per tensor.
    lay = mamba_param_layout()
    m_init = _build_eager_mamba(seed=123).to(dev)
    W = MambaWeights.from_named({n: p.detach().double()
                                 for n, p in m_init.named_parameters()})
    _, cache = mamba_forward(W, tokens.to(dev), targets.to(dev))
    ograds = mamba_backward(W, cache)
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
        f"— the hand-written selective-scan backward diverges from the reference.")
    # (a.3) params after 1 step: rel 1e-5 with an ABSOLUTE floor of 0.05*lr.
    # Why the floor (measured on the decoder, the SAME phenomenon — see
    # test_decoder_l3_real_single_step_parity a.3): at step 1 AdamW's update is
    # lr*g/(|g|+eps). For elements whose TRUE grad is ~0, kernel and eager both
    # hold fp32 reduction noise |g|~1e-10 (the fp64 oracle confirms ~0) and the
    # update becomes lr*g/eps: a dg~5e-10 ORDERING difference between two correct
    # fp32 reductions yields |dp| up to lr*5e-10/1e-8 = 0.05*lr. That is reference
    # noise, not kernel error; the pure-rel gate also collapses on zero-init
    # tensors. Systematic apply errors are caught by the 200-step trajectory test.
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
@requires_mamba_l3
def test_mamba_l3_real_trajectory():
    """(b) 200-step loss curve tracks eager AdamW (1e-3 rel); final params
    chaos-calibrated (see below). MIRRORS test_decoder_l3_real_trajectory."""
    import torch.nn.functional as F
    from tests.hw.mamba_oracle import VOCAB, P_HEAD, SEQ
    dev = "cuda"
    m_eager = _build_eager_mamba(seed=321).to(dev)
    m_fused = _build_eager_mamba(seed=321).to(dev)
    m_fused.load_state_dict(m_eager.state_dict())
    B = 512
    g = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=g)
    targets = torch.randint(0, P_HEAD, (B,), generator=g)
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
        lf = _run_mamba_l3_real_step(m_fused, tokens, targets, lr, betas, wd,
                                     eps, t, state, dev)
        rel = abs(lf - le.item()) / (abs(le.item()) + 1e-30)
        max_loss_dev = max(max_loss_dev, rel)
    # Loss curve agreement is the SHARP gate (a systematically wrong apply or
    # backward shifts the curve immediately; it must hold <1e-3 across all 200
    # steps). Final params get a CHAOS-CALIBRATED gate: two correct fp32
    # implementations separate chaotically on this loss surface — the decoder's
    # MEASURED control (eager-vs-eager with a single 1e-7 rel perturbation on ONE
    # tensor separating to 5.9e-3 over 200 steps; kernel-vs-eager 7.0e-3) sets the
    # 2e-2 ≈ 3x-chaos-floor gate; a real formula error (e.g. inverted weight decay)
    # compounds far beyond it AND trips the loss gate. The SHARP loss gate is the
    # real correctness check here.
    assert max_loss_dev < 1e-3, f"max loss-traj rel dev {max_loss_dev:.2e} > 1e-3"
    worst_p = 0.0
    ep = dict(m_eager.named_parameters())
    for n, p in m_fused.named_parameters():
        a = (p.detach() - ep[n].detach()).abs().max().item()
        denom = ep[n].detach().abs().max().item() + 1e-30
        worst_p = max(worst_p, a / denom)
    assert worst_p < 2e-2, (
        f"final params rel {worst_p:.2e} > 2e-2 (3x the measured fp32 chaos "
        f"floor — see the decoder control experiment referenced above)")


@pytest.mark.hw
@requires_mamba_l3
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mamba_l3_real_groks(seed):
    """(c) 3-seed grok smoke through the race's Mamba train path routing the cell
    through the L3-REAL megakernel. MIRRORS test_decoder_l3_real_groks. Skipped
    cleanly until the Mamba grok-smoke driver is wired (the decoder's
    _grok_smoke_impl analogue)."""
    try:
        from tests.hw._mamba_grok_smoke import mamba_grok_smoke_impl
    except Exception:
        pytest.skip("Mamba grok-smoke driver not wired yet (see INTEGRATION-MAMBA.md)")
    best = mamba_grok_smoke_impl(seed)
    assert best >= 0.95, (
        f"(mamba, adamw) did not grok with the L3-REAL fused train step "
        f"(seed {seed}): best test acc {best:.3f} < 0.95")
