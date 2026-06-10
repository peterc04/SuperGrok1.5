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


requires_mamba_l3 = pytest.mark.skipif(
    not _have_mamba_l3_real(),
    reason="L3-REAL Mamba path unavailable (needs sm_90 + the composed launcher "
           "+ dispatch wiring; built in a later phase).")


@pytest.mark.hw
@requires_mamba_l3
def test_mamba_l3_real_single_step_parity():
    """(a) single-step parity vs eager: kernel loss within 1e-5 rel, every weight
    grad within 1e-4 rel (vs the oracle), params after 1 step 1e-5 rel.

    MIRRORS test_decoder_l3_real_single_step_parity. The keystone is the
    PER-TENSOR grad comparison of the kernel's reduced weight grad against the
    oracle — it exercises the hand-written selective-scan backward's magnitudes.
    Implemented when the Mamba composition/launcher + dispatch wiring land."""
    pytest.skip("Mamba L3-REAL launcher/dispatch not wired in this phase "
                "(see INTEGRATION-MAMBA.md); CPU oracle+mirror cover the math.")


@pytest.mark.hw
@requires_mamba_l3
def test_mamba_l3_real_trajectory():
    """(b) 200-step loss curve tracks eager AdamW (1e-3 rel); final params 1e-3
    rel (fp32 accumulation drift). MIRRORS test_decoder_l3_real_trajectory."""
    pytest.skip("Mamba L3-REAL launcher/dispatch not wired in this phase.")


@pytest.mark.hw
@requires_mamba_l3
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mamba_l3_real_groks(seed):
    """(c) 3-seed grok smoke through train_adamw routing the Mamba cell through
    the L3-REAL megakernel. MIRRORS test_decoder_l3_real_groks."""
    pytest.skip("Mamba L3-REAL launcher/dispatch not wired in this phase.")
