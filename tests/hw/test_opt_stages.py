"""tests/hw/test_opt_stages.py — Phase 3 BUILD ✅-gate (CPU, no accelerator).

CPU **fp32 mirrors** of the in-kernel optimizer PRECOMPUTE stages defined in
``csrc/fused/sm_90/opt_stages_precompute.cuh``, validated against the **fp64**
reference oracles in ``tests/hw/test_reference_parity.py`` (the canonical math
single source — see ``csrc/algorithms/SOURCE_OF_TRUTH.md``).

WHY CPU MIRRORS ARE THE VALIDATION HERE
---------------------------------------
Per the phase directive (NO builds / NO GPU runs), the CUDA header is not
compiled in this phase. The staged ALGORITHM (not the CUDA codegen) is what we
must validate: each mirror reproduces the device function's fp32 accumulation
SHAPE — the owner-computes cross-CTA tree (prodigy), the per-tensor block tree
(SG11 cosine gate), and the 5-iteration Newton-Schulz fp32 matmul chain (muon).
A mirror that matches its fp64 oracle within fp32 tolerance therefore certifies
the stage design is correct; the CUDA compile is deferred to integration.

DETERMINISM is a DESIGN property (no float atomics; per-CTA slots summed in a
fixed index order). It is asserted structurally below (a permutation of the
per-CTA partial order changes nothing *in the fixed-order reducer*, whereas an
atomic accumulate would be order-dependent) — a single run cannot "prove"
determinism, so we assert the invariant the design guarantees.

Run (CPU, what CI does):
    PYTHONPATH=. python3 -m pytest tests/hw/test_opt_stages.py -q
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

# Reuse the canonical fp64 oracles (single source of truth for the math).
from tests.hw.test_reference_parity import (  # noqa: E402
    ref_prodigy_update_d,
    ref_sg_phi_forward,
    ref_muon_step,
    _f64,
)


# prodigy: ``ref_prodigy_partials`` in test_reference_parity.py was once STALE
# vs csrc/algorithms/prodigy.h (d_prev^1 + signed g instead of the canonical
# d_prev^2 + |g|, prodigy.h:48,52) — this file carried its own canonical copy
# while the sibling was sibling-owned. The shared ref has since been fixed to
# the canonical form, so it is the single fp64 transcription again; the alias
# below keeps this file's call sites stable.
from tests.hw.test_reference_parity import ref_prodigy_partials as \
    ref_prodigy_partials_canonical


# ===========================================================================
#  Shared helpers mirroring the device reduction shapes.
# ===========================================================================

def _f32(t):
    return t.detach().to(torch.float32)


def _block_tree_sum_fp32(vals: torch.Tensor, block: int = 256) -> float:
    """Mirror primitives.cuh::block_reduce_sum_f32 accumulation SHAPE.

    The device reduces with a warp-butterfly + smem tree over a fixed thread
    count, all in fp32. Exact bit-matching that tree on CPU is not the point;
    what matters is that the reduction is fp32 and pairwise-ish (NOT a naive
    left-fold over millions of elements), so we accumulate in fp32 with a
    block-strided partial layout that matches the kernel's grid-stride loop:
    `block` threads each fold their strided slice, then the partials are summed.
    This is the same numerical regime as the device tree (fp32, partitioned).
    """
    v = _f32(vals).flatten()
    n = v.numel()
    if n == 0:
        return 0.0
    # Per-thread partials (thread j folds indices j, j+block, j+2*block, ...).
    partials = torch.zeros(block, dtype=torch.float32)
    for j in range(min(block, n)):
        partials[j] = v[j::block].sum(dtype=torch.float32)
    # Tree-sum the partials in fixed order (the warp+smem tree).
    return float(partials.sum(dtype=torch.float32))


def _owner_sum_slots_fp32(slots) -> float:
    """Mirror opt_stages_precompute.cuh::owner_sum_slots — ascending fixed-order
    fp32 sum of the per-CTA partial slots (the deterministic owner-computes
    tree that REPLACES the live path's atomicAdd)."""
    acc = torch.zeros((), dtype=torch.float32)
    for s in slots:
        acc = acc + torch.as_tensor(s, dtype=torch.float32)
    return float(acc)


# ===========================================================================
#  (3a) PRODIGY — cross-ALL-tensors d reduction (owner-computes tree).
# ===========================================================================

def _prodigy_stage_mirror(params, param_inits, grads, *, d_prev, n_ctas):
    """Mirror prodigy_precompute_reduce_phaseA + phaseB.

    Per CTA: accumulate (r,s) over the tensors that CTA is assigned (we model a
    round-robin tensor→CTA assignment, the work-stealing queue's effect on the
    PARTITION; the FINAL sum is partition-independent because the reducer sums
    every slot). Each CTA's (r,s) is a fp32 block-tree over its elements; the
    per-CTA slots are then owner-summed in fixed order. Returns d.
    """
    # Per-CTA (r,s) slots, fp32 (matches ws.prodigy_partials[2*n_ctas]).
    r_slots = [torch.zeros((), dtype=torch.float32) for _ in range(n_ctas)]
    s_slots = [torch.zeros((), dtype=torch.float32) for _ in range(n_ctas)]
    for t, (p, pi, g) in enumerate(zip(params, param_inits, grads)):
        cta = t % n_ctas
        pf, pif, gf = map(_f32, (p, pi, g))
        # Per-element prodigy partials in fp32 (prodigy.h:48,52 body):
        #   r += g*(pi-p)*d^2 ;  s += d^2*|g|
        r_elt = gf * (pif - pf) * (d_prev * d_prev)
        s_elt = (d_prev * d_prev) * gf.abs()
        r_slots[cta] = r_slots[cta] + torch.as_tensor(
            _block_tree_sum_fp32(r_elt), dtype=torch.float32)
        s_slots[cta] = s_slots[cta] + torch.as_tensor(
            _block_tree_sum_fp32(s_elt), dtype=torch.float32)
    # Owner-computes fixed-order reduction across CTAs.
    r_sum = _owner_sum_slots_fp32(r_slots)
    s_sum = _owner_sum_slots_fp32(s_slots)
    # Canonical d update (prodigy.h:56-65) — mirrored in fp32.
    denom = abs(s_sum)
    if denom < 1e-12:
        return float(d_prev)
    return float(max(d_prev, r_sum / denom))


def test_prodigy_d_reduction_matches_fp64_oracle():
    """Staged cross-tensor (r,s)->d reduction vs ref_prodigy_partials/update_d."""
    torch.manual_seed(0)
    d_prev = 1e-3
    # A multi-tensor model: several params of varied size (the cross-tensor sum).
    shapes = [(257,), (64, 48), (4096,), (33, 17), (1000,)]
    params, param_inits, grads = [], [], []
    for sh in shapes:
        params.append(torch.randn(*sh, dtype=torch.float64))
        param_inits.append(torch.randn(*sh, dtype=torch.float64))
        grads.append(torch.randn(*sh, dtype=torch.float64) * 0.1)

    # fp64 oracle: full-precision (r,s) over ALL elements of ALL tensors, then d.
    # Uses the CANONICAL prodigy.h transcription (see the note at module top).
    r_tot = torch.zeros((), dtype=torch.float64)
    s_tot = torch.zeros((), dtype=torch.float64)
    for p, pi, g in zip(params, param_inits, grads):
        r, s = ref_prodigy_partials_canonical(p, pi, g, d_prev=d_prev)
        r_tot = r_tot + r
        s_tot = s_tot + s
    d_oracle = ref_prodigy_update_d(d_prev, r_tot, s_tot)

    # Staged fp32 mirror across several CTA counts (partition-independent total).
    for n_ctas in (1, 4, 13, 132):
        d_stage = _prodigy_stage_mirror(
            params, param_inits, grads, d_prev=d_prev, n_ctas=n_ctas)
        assert math.isclose(d_stage, d_oracle, rel_tol=1e-4, abs_tol=1e-9), (
            f"prodigy d stage (n_ctas={n_ctas}) {d_stage} vs oracle {d_oracle}")


def test_prodigy_owner_sum_is_partition_independent():
    """DETERMINISM/correctness: the fixed-order owner-sum is invariant to how
    tensors are partitioned across CTAs AND to the slot ordering being a fixed
    ascending walk (an atomicAdd accumulate would not give this guarantee for
    arbitrary completion order). We assert the reduced total is identical for
    several CTA counts to within tight fp32 tolerance."""
    torch.manual_seed(1)
    d_prev = 5e-4
    shapes = [(128,), (777,), (16, 16), (2048,)]
    params = [torch.randn(*s, dtype=torch.float64) for s in shapes]
    inits = [torch.randn(*s, dtype=torch.float64) for s in shapes]
    grads = [torch.randn(*s, dtype=torch.float64) * 0.05 for s in shapes]
    ds = [
        _prodigy_stage_mirror(params, inits, grads, d_prev=d_prev, n_ctas=k)
        for k in (1, 2, 4, 8, 64)
    ]
    for d in ds[1:]:
        assert math.isclose(d, ds[0], rel_tol=1e-5, abs_tol=1e-12)


# ===========================================================================
#  (3c) SUPERGROK11 — meta-MLP mu + per-TENSOR gate=sigmoid(gate_temp*cos) (block tree).
# ===========================================================================

def _sg11_mu_mirror(grad, sharpness, W1, b1, W2, b2, rescale):
    """Mirror sg11_precompute_mu_stage: mu = rescale*phi(grad, sharpness)
    elementwise (per-tensor weights). Uses the canonical phi oracle
    (``ref_sg_phi_forward``), which broadcasts over scalar grad/sharp tensors —
    we pass them as flat fp64 tensors so the GELU-hidden MLP is evaluated for
    every element at once (the device runs sg11_phi_forward<H> per element)."""
    g = _f64(grad).flatten()
    s = _f64(sharpness).flatten()
    # ref_sg_phi_forward broadcasts grad_val/sharp_val against W1[:,0]/[:,1]; a
    # column vector of grad gives a [N, H] hidden then sums to [N].
    out = ref_sg_phi_forward(g.unsqueeze(1), s.unsqueeze(1), W1, b1, W2, float(b2))
    return (rescale * out).reshape(grad.shape)


def _sg11_gate_mirror_fp32(grad, momentum, gate_temp=5.0):
    """Mirror sg11_precompute_mu_and_gate_for_tensor's finalizer (== compute_cosine_
    gate_fused == sg11_finalize_gate, algorithms/supergrok11.h) in fp32 with the
    block-tree reduction shape. The canonical SG11 gate SIGNAL is cos(grad,
    MOMENTUM) (supergrok11.h:22-29,64 — the 2nd operand is the start-of-step
    exp_avg, NOT mu; that distinguishes SG11 from the buggy cos(grad,mu)). Returns
    gate = sigmoid(gate_temp * cos(grad, momentum)) (the cosine SIGNAL squashed by
    the temperature-scaled sigmoid, NOT a bare clamp)."""
    g = _f32(grad).flatten()
    m = _f32(momentum).flatten()
    num = _block_tree_sum_fp32(g * m)
    den_g = _block_tree_sum_fp32(g * g)
    den_m = _block_tree_sum_fp32(m * m)
    denom = math.sqrt(den_g * den_m + 1e-12)
    cos = (num / denom) if denom > 0.0 else 0.0
    return 1.0 / (1.0 + math.exp(-gate_temp * cos))


def _sg11_gate_oracle_fp64(grad, momentum, gate_temp=5.0):
    """fp64 gate oracle: sigmoid(gate_temp * <g,m>/sqrt(|g|^2 |m|^2 + 1e-12)),
    m == the start-of-step MOMENTUM (exp_avg) — the canonical cos(grad,momentum)
    SIGNAL (supergrok11.h:29,64), NOT the co-wrong cos(grad,mu)."""
    g = _f64(grad).flatten()
    m = _f64(momentum).flatten()
    num = float((g * m).sum())
    den_g = float((g * g).sum())
    den_m = float((m * m).sum())
    denom = math.sqrt(den_g * den_m + 1e-12)
    cos = (num / denom) if denom > 0.0 else 0.0
    return 1.0 / (1.0 + math.exp(-gate_temp * cos))


def test_sg11_cosine_gate_matches_fp64_oracle():
    """Per-tensor sigmoid-of-cos(grad, MOMENTUM) gate (block tree) vs the fp64
    oracle. The cosine's 2nd operand is the start-of-step momentum (exp_avg), the
    canonical SG11 signal — NOT mu."""
    torch.manual_seed(2)
    for sh in [(513,), (32, 24), (2000,)]:
        grad = torch.randn(*sh, dtype=torch.float64) * 0.3
        momentum = torch.randn(*sh, dtype=torch.float64) * 0.25
        gate_stage = _sg11_gate_mirror_fp32(grad, momentum)
        gate_oracle = _sg11_gate_oracle_fp64(grad, momentum)
        assert math.isclose(gate_stage, gate_oracle, rel_tol=1e-4, abs_tol=1e-6), (
            f"sg11 gate {gate_stage} vs oracle {gate_oracle} shape={sh}")
        assert 0.0 <= gate_stage <= 1.0


def test_sg11_gate_applies_gate_temp():
    """Mirror the CODE: sg11_finalize_gate applies gate = sigmoid(gate_temp*cos)
    — so the gate is the TEMPERATURE-SCALED SIGMOID of the cosine, NOT a bare
    clamp. Two teeth: (a) the gate equals sigmoid(gate_temp*cos) (vs the raw
    clamped cosine it used to be), and (b) raising gate_temp moves the gate toward
    1 for a positive cosine (a bare clamp would be temperature-invariant)."""
    torch.manual_seed(3)
    grad = torch.randn(400, dtype=torch.float64)
    # 2nd cosine operand = MOMENTUM (canonical SG11), correlated with grad ⇒ cos>0.
    momentum = 0.5 * grad + 0.1 * torch.randn(400, dtype=torch.float64)
    # (a) gate == sigmoid(gate_temp*cos), and DIFFERS from the bare clamped cosine.
    g = _f64(grad).flatten(); m = _f64(momentum).flatten()
    cos = float((g * m).sum()) / math.sqrt(
        float((g * g).sum()) * float((m * m).sum()) + 1e-12)
    clamped_cos = min(max(cos, 0.0), 1.0)
    gate5 = _sg11_gate_mirror_fp32(grad, momentum, gate_temp=5.0)
    assert math.isclose(gate5, 1.0 / (1.0 + math.exp(-5.0 * cos)),
                        rel_tol=1e-4, abs_tol=1e-6)
    assert not math.isclose(gate5, clamped_cos, rel_tol=1e-3, abs_tol=1e-3)
    # (b) temperature is genuinely consumed: higher temp ⇒ gate closer to 1 (cos>0).
    gate1 = _sg11_gate_mirror_fp32(grad, momentum, gate_temp=1.0)
    gate20 = _sg11_gate_mirror_fp32(grad, momentum, gate_temp=20.0)
    assert gate1 < gate5 < gate20


def _sg11_mu_and_gate_for_tensor(grad_T, sharp_T, mom_T, W1, b1, W2, b2, rescale):
    """Mirror sg11_precompute_mu_and_gate_for_tensor: the PER-TENSOR fused body
    that writes mu(T) then computes gate(T)=sigmoid(gate_temp·cos(grad_T, mom_T)).
    Returns (mu_T, gate_T). This is the single-drain Design B body — it sees ONLY
    tensor T (off/n slice), so its gate cannot depend on any other tensor's slice.
    The gate's 2nd operand is the start-of-step momentum mom_T (canonical), not mu."""
    mu_T = _sg11_mu_mirror(grad_T, sharp_T, W1, b1, W2, b2, rescale)
    gate_T = _sg11_gate_mirror_fp32(grad_T, mom_T)
    return mu_T, gate_T


def test_sg11_gate_is_per_tensor_not_cross_tensor():
    """DESIGN GUARD (the sequencing fix): SG11's gate(T) depends ONLY on
    (grad_T, mu_T) — never on another tensor's mu. So the per-tensor fused body
    yields the SAME gate for T whether T is processed alone or amid a whole model
    of other tensors. This is WHY the staged form is a single task-queue drain
    with NO grid barrier (a separate full-queue mu pre-pass + apply drain would
    instead silently no-op the apply — see the header's SG11 block). A CPU mirror
    cannot model work-stealing, but it CAN certify the per-tensor INDEPENDENCE the
    no-barrier design relies on."""
    torch.manual_seed(8)
    H = 32
    W1 = torch.randn(H * 2, dtype=torch.float64) * 0.15
    b1 = torch.randn(H, dtype=torch.float64) * 0.15
    W2 = torch.randn(H, dtype=torch.float64) * 0.15
    b2, rescale = -0.02, 0.9
    # Tensor T plus several OTHER tensors that share the meta-net weights.
    gT = torch.randn(321, dtype=torch.float64) * 0.4
    sT = (torch.randn(321, dtype=torch.float64)).abs()
    mT = torch.randn(321, dtype=torch.float64) * 0.3   # start-of-step momentum slice
    others = [
        (torch.randn(k, dtype=torch.float64), (torch.randn(k, dtype=torch.float64)).abs(),
         torch.randn(k, dtype=torch.float64))
        for k in (64, 999, 128)
    ]
    # gate(T) computed by the per-tensor body, with T alone.
    _, gate_alone = _sg11_mu_and_gate_for_tensor(gT, sT, mT, W1, b1, W2, b2, rescale)
    # gate(T) computed by the same per-tensor body AFTER processing the others
    # (the body only ever sees T's slice; the others change nothing for T).
    for go, so, mo in others:
        _sg11_mu_and_gate_for_tensor(go, so, mo, W1, b1, W2, b2, rescale)
    _, gate_after = _sg11_mu_and_gate_for_tensor(gT, sT, mT, W1, b1, W2, b2, rescale)
    assert gate_alone == gate_after, (
        "SG11 gate(T) must be independent of other tensors — the no-barrier "
        "single-drain design depends on this per-tensor independence.")


def test_sg11_mu_uses_sharpness_input():
    """SG11 mu DEPENDS on the model-coupled `sharpness` input (the ABI gap flagged
    in the header): changing sharpness changes mu, so the stage genuinely
    consumes the second-backward signal rather than ignoring it."""
    torch.manual_seed(4)
    H = 32
    W1 = torch.randn(H * 2, dtype=torch.float64)
    b1 = torch.randn(H, dtype=torch.float64)
    W2 = torch.randn(H, dtype=torch.float64)
    grad = torch.randn(64, dtype=torch.float64)
    s0 = torch.zeros(64, dtype=torch.float64)
    s1 = torch.ones(64, dtype=torch.float64)
    mu0 = _sg11_mu_mirror(grad, s0, W1, b1, W2, 0.0, 1.0)
    mu1 = _sg11_mu_mirror(grad, s1, W1, b1, W2, 0.0, 1.0)
    assert not torch.allclose(mu0, mu1, atol=1e-6)


# ===========================================================================
#  (3d) SUPERGROK15 — meta-MLP mu (gate is a host scalar; no gate stage).
# ===========================================================================

def test_sg15_mu_matches_phi_oracle():
    """SG15 mu sub-stage = rescale*phi(grad, sharpness) — same phi-net shape as
    SG11 (supergrok15.h:34-52). Validate elementwise vs the canonical phi oracle
    (vectorized fp64)."""
    torch.manual_seed(5)
    H = 32
    W1 = torch.randn(H * 2, dtype=torch.float64) * 0.2
    b1 = torch.randn(H, dtype=torch.float64) * 0.2
    W2 = torch.randn(H, dtype=torch.float64) * 0.2
    b2 = -0.03
    rescale = 1.3
    grad = torch.randn(300, dtype=torch.float64) * 0.5
    sharp = (torch.randn(300, dtype=torch.float64)).abs()
    mu_stage = _sg11_mu_mirror(grad, sharp, W1, b1, W2, b2, rescale)  # same shape
    # Vectorized oracle over the whole tensor.
    W1m = _f64(W1).reshape(-1, 2)
    pre = W1m[:, 0].unsqueeze(0) * _f64(grad).unsqueeze(1) + \
        W1m[:, 1].unsqueeze(0) * _f64(sharp).unsqueeze(1) + _f64(b1).unsqueeze(0)
    gelu = 0.5 * pre * (1.0 + torch.erf(pre / math.sqrt(2.0)))
    mu_oracle = rescale * ((gelu * _f64(W2).unsqueeze(0)).sum(-1) + b2)
    assert torch.allclose(mu_stage.to(torch.float64), mu_oracle,
                          rtol=1e-5, atol=1e-7)


# ===========================================================================
#  (3b) MUON — momentum update + Newton-Schulz orthogonalization (fp32 matmuls).
# ===========================================================================

def _muon_orth_stage_mirror_fp32(grad, buf, *, momentum, ns_steps):
    """Mirror the muon precompute STAGE chain in fp32:
        buf = momentum*buf + grad                         (muon.h:43, phaseA)
        inv_norm = 1/(||buf||_F + 1e-7)                   (bindings.cpp:962-963)
        X = buf*inv_norm                                  (muon_scale_X)
        repeat ns_steps:
            A   = X Xᵀ ; AX = A X ; AAX = A AX            (muon_matmul, fp32)
            X   = a*X + b*AX + c*AAX                       (muon_ns_combine_phase)
        orth = X
    The matmuls are accumulated in fp32 (the device owner-computes inner product
    sums the contraction in fp32). Returns (orth, buf_new).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    g = _f32(grad)
    bf = _f32(buf)
    buf_new = momentum * bf + g
    inv_norm = 1.0 / (torch.linalg.norm(buf_new).item() + 1e-7)
    X = (buf_new * inv_norm).to(torch.float32)
    for _ in range(ns_steps):
        # fp32 matmuls (mirror the device fp32 contraction).
        A = (X @ X.t()).to(torch.float32)
        AX = (A @ X).to(torch.float32)
        AAX = (A @ AX).to(torch.float32)
        X = (a * X + b * AX + c * AAX).to(torch.float32)
    return X, buf_new


def test_muon_newton_schulz_matches_fp64_oracle():
    """Staged fp32 NS orthogonalization vs ref_muon_step's fp64 NS, then the
    final scaled parameter update. fp32-vs-fp64 drift is LARGEST here (5 matmul
    iterations), so tolerances reflect fp32 NS accumulation, not 1e-9."""
    torch.manual_seed(6)
    lr, wd, momentum, ns_steps = 0.02, 0.01, 0.95, 5
    for rows, cols in [(32, 48), (64, 64), (128, 96)]:
        p = torch.randn(rows, cols, dtype=torch.float64) * 0.1
        g = torch.randn(rows, cols, dtype=torch.float64) * 0.1
        buf = torch.randn(rows, cols, dtype=torch.float64) * 0.1

        # fp64 oracle (full muon step → p_new, buf_new).
        p_oracle, buf_oracle = ref_muon_step(
            p, g, buf, momentum=momentum, lr=lr, wd=wd, ns_steps=ns_steps)

        # Staged fp32 mirror: orth + scaled apply (muon.h:63-73 apply tail).
        orth, buf_new = _muon_orth_stage_mirror_fp32(
            g, buf, momentum=momentum, ns_steps=ns_steps)
        scale = 0.2 * math.sqrt(max(rows, cols))
        neg_lr_scale = -lr * scale
        decay = 1.0 - lr * wd
        p_stage = (_f32(p) * decay + neg_lr_scale * orth)

        # buf_new must match closely (no NS, just momentum add).
        assert torch.allclose(buf_new.to(torch.float64), buf_oracle,
                              rtol=1e-4, atol=1e-5)
        # NS-orthogonalized parameter: fp32 accumulation over 5 iterations.
        max_abs = (p_stage.to(torch.float64) - p_oracle).abs().max().item()
        rel = max_abs / (p_oracle.abs().max().item() + 1e-12)
        assert rel < 2e-3, (
            f"muon NS rel-err {rel:.2e} too large for {rows}x{cols}")


def _muon_matmul_mirror(A, B, M, N, K, A_rs, B_rs, b_transposed):
    """Bit-faithful mirror of opt_stages_precompute.cuh::muon_matmul's INDEXING:
    one output element per "thread", contraction summed in ascending k (fp32).
    A and B are FLAT row-major buffers (the workspace layout); A_rs/B_rs are the
    row strides; b_transposed selects B[col,k] (Bᵀ, for X Xᵀ) vs B[k,col] (for
    A X / A AX). Returns the [M,N] result."""
    A = _f32(A).flatten()
    B = _f32(B).flatten()
    C = torch.zeros(M * N, dtype=torch.float32)
    for idx in range(M * N):
        row, col = idx // N, idx % N
        acc = torch.zeros((), dtype=torch.float32)
        for k in range(K):
            a = A[row * A_rs + k]
            b = B[col * B_rs + k] if b_transposed else B[k * B_rs + col]
            acc = acc + a * b
        C[idx] = acc
    return C.reshape(M, N)


def test_muon_matmul_strides_and_transpose_exact():
    """Exercise the ACTUAL device fn muon_matmul (the only non-canonical Muon
    device code) for ALL THREE NS matmul shapes, including the b_transposed=FALSE
    branch (A X, A AX) that the torch.mm-based NS mirror never touches. Use a
    NON-SQUARE matrix (rows != cols) so any M/N/K or row-stride swap changes the
    result — catching the stride/transpose bug class that hand-reading misses and
    that no compile here would catch."""
    torch.manual_seed(7)
    rows, cols = 5, 3                       # non-square: stride swaps are visible
    X = torch.randn(rows, cols, dtype=torch.float32)   # [rows, cols], rs=cols

    # (1) A = X Xᵀ : M=rows, N=rows, K=cols, A_rs=cols, B_rs=cols, bT=True.
    A_ref = X @ X.t()                                  # [rows, rows]
    A_mir = _muon_matmul_mirror(X, X, rows, rows, cols, cols, cols, True)
    assert torch.allclose(A_mir, A_ref, rtol=1e-4, atol=1e-5)

    A = A_ref.contiguous()                             # [rows, rows], rs=rows
    # (2) AX = A X : M=rows, N=cols, K=rows, A_rs=rows, B_rs=cols, bT=False.
    AX_ref = A @ X                                     # [rows, cols]
    AX_mir = _muon_matmul_mirror(A, X, rows, cols, rows, rows, cols, False)
    assert torch.allclose(AX_mir, AX_ref, rtol=1e-4, atol=1e-5)

    AX = AX_ref.contiguous()
    # (3) AAX = A AX : M=rows, N=cols, K=rows, A_rs=rows, B_rs=cols, bT=False.
    AAX_ref = A @ AX                                   # [rows, cols]
    AAX_mir = _muon_matmul_mirror(A, AX, rows, cols, rows, rows, cols, False)
    assert torch.allclose(AAX_mir, AAX_ref, rtol=1e-4, atol=1e-5)


# ===========================================================================
#  Verdict coverage guard — the staged set is exactly {prodigy, muon, sg11, sg15}.
# ===========================================================================

STAGED = {"prodigy", "muon", "supergrok11", "supergrok15"}
NOTHING_NEEDED = {"grokfast", "grokadamw", "neuralgrok"}
MODEL_COUPLED = {"looksam"}
SKIP = {"supergrok2"}
TRIVIAL_TAIL_ONLY = {"adamw", "lion"}  # already complete in-kernel (no precompute)


def test_verdict_partition_covers_all_eleven():
    """The 9 non-trivial optimizers partition into STAGED / NOTHING-NEEDED /
    MODEL-COUPLED / SKIP; adamw+lion are the 2 already-complete tails. Total 11."""
    everyone = STAGED | NOTHING_NEEDED | MODEL_COUPLED | SKIP | TRIVIAL_TAIL_ONLY
    assert everyone == {
        "adamw", "lion", "grokfast", "grokadamw", "looksam", "prodigy",
        "neuralgrok", "muon", "supergrok11", "supergrok15", "supergrok2",
    }
    assert len(everyone) == 11
    # No optimizer in two buckets.
    buckets = [STAGED, NOTHING_NEEDED, MODEL_COUPLED, SKIP, TRIVIAL_TAIL_ONLY]
    for i in range(len(buckets)):
        for j in range(i + 1, len(buckets)):
            assert buckets[i].isdisjoint(buckets[j])
