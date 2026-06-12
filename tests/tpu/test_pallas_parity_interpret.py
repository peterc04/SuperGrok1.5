"""tests/tpu/test_pallas_parity_interpret.py — Pallas optimizer parity (interpret).

FIRST functional execution of the TPU-v6e Pallas optimizer kernels. Each of the
11 race optimizers' per-tensor ``launch_<opt>_step`` (the fused-TU contract math
in ``csrc/backends/pallas/launch_<opt>.py`` — the canonical TPU source per
``csrc/algorithms/SOURCE_OF_TRUTH.md``) is wrapped in a whole-array
``pl.pallas_call(..., interpret=True)`` and run on CPU against the fp64-accumulate
reference functions in ``tests/hw/test_reference_parity.py`` (transcribed 1:1 from
``csrc/algorithms/<opt>.h``).

Design (deliberately a MATH-parity harness, not a tiling harness):
  * ONE block over the whole array (BlockSpec defaults), so the test isolates the
    per-element optimizer math and is immune to tiling/pipelining bugs. Odd /
    non-multiple-of-8 shapes are still exercised via arbitrary array sizes; TPU
    tiling correctness is a silicon-day-1 item (see the day-1 checklist in the
    audit report / module docstring of csrc/fused/tpu_v6e).
  * fp32 kernel inputs vs the fp64 reference, at the SAME fp32-vs-fp64 tolerance
    discipline the GPU half of test_reference_parity.py uses (atol=1e-4,
    rtol=1e-3 — see test_kernel_matches_reference_gpu). 1e-9/1e-7 there is the
    fp64-vs-fp64 self-consistency band and would be wrong here.
  * Multiple sizes, seeds, NONZERO initial state, and several sequential steps for
    the stateful EMA optimizers (state threaded through both paths in lockstep).

Skipped cleanly when JAX is absent so CI stays green:
    PYTHONPATH=. python3 -m pytest tests/tpu/test_pallas_parity_interpret.py -q
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")           # whole suite skips on a torch-only runner
import jax.numpy as jnp                     # noqa: E402
import numpy as np                          # noqa: E402
from jax.experimental import pallas as pl   # noqa: E402

torch = pytest.importorskip("torch")        # the fp64 references are PyTorch

# fp64-accumulate reference implementations (transcribed from csrc/algorithms/*.h).
from tests.hw.test_reference_parity import (  # noqa: E402
    ref_adamw_step,
    ref_lion_step,
    ref_grokadamw_step,
    ref_grokfast_step,
    ref_neuralgrok_step,
    ref_neuralgrok_psi_forward,
    ref_looksam_apply_step,
    ref_looksam_perturb,
    ref_looksam_set_direction,
    ref_prodigy_apply_step,
    ref_prodigy_partials,
    ref_prodigy_update_d,
    ref_sg11_step,
    ref_sg15_step,
    ref_sg_phi_forward,
    ref_sg2_apply_step,
    ref_muon_step,
)

# The Pallas (canonical TPU) optimizer launchers under test.
from csrc.backends.pallas.launch_adamw import launch_adamw_step
from csrc.backends.pallas.launch_lion import launch_lion_step
from csrc.backends.pallas.launch_grokadamw import launch_grokadamw_step
from csrc.backends.pallas.launch_grokfast import launch_grokfast_step
from csrc.backends.pallas.launch_neuralgrok import launch_neuralgrok_step
from csrc.backends.pallas.launch_looksam import (
    launch_looksam_apply,
    launch_looksam_perturb,
    launch_looksam_set_direction,
)
from csrc.backends.pallas.launch_prodigy import launch_prodigy_step
from csrc.backends.pallas.launch_supergrok11 import launch_supergrok11_step
from csrc.backends.pallas.launch_supergrok15 import launch_supergrok15_step
from csrc.backends.pallas.launch_supergrok2 import launch_supergrok2_step


# fp32-vs-fp64 tolerance (matches the GPU half of test_reference_parity.py).
ATOL = 1e-4
RTOL = 1e-3

# Sizes: include odd, prime, and non-multiple-of-8 lengths.
SIZES = [7, 13, 16, 31, 64, 100]
SEEDS = [0, 1, 7]


# ---------------------------------------------------------------------------
#  Interpret-mode harness: wrap a pure-JAX launch fn in pl.pallas_call.
# ---------------------------------------------------------------------------

def _run_interp(fn, array_args, scalar_kwargs, n_out):
    """Execute ``fn`` inside a whole-array ``pl.pallas_call(interpret=True)``.

    ``array_args``  : tuple of jnp arrays passed positionally (first, in order).
    ``scalar_kwargs``: dict of python-scalar kwargs closed over by the kernel.
    ``n_out``       : number of array outputs ``fn`` returns.

    The kernel body reads every input ref whole, calls ``fn``, and writes each
    output ref whole. A single block (default BlockSpec) covers the whole array,
    so this validates math, not tiling.
    """
    out_shapes = tuple(
        jax.ShapeDtypeStruct(array_args[0].shape, jnp.float32)
        for _ in range(n_out)
    )

    def kernel(*refs):
        in_refs = refs[:len(array_args)]
        out_refs = refs[len(array_args):]
        vals = tuple(r[...] for r in in_refs)
        res = fn(*vals, **scalar_kwargs)
        if not isinstance(res, tuple):
            res = (res,)
        for o_ref, val in zip(out_refs, res):
            o_ref[...] = val.astype(jnp.float32)

    outs = pl.pallas_call(
        kernel,
        out_shape=out_shapes if n_out > 1 else out_shapes[0],
        interpret=True,
    )(*array_args)
    if n_out == 1:
        outs = (outs,)
    return tuple(np.asarray(o, dtype=np.float64) for o in outs)


def _t(np_arr):
    """numpy float64 -> torch float64 (reference input). Copy so the tensor is
    writable (np.asarray on a jax->np result can be read-only)."""
    return torch.from_numpy(
        np.array(np_arr, dtype=np.float64, copy=True)).to(torch.float64)


def _close(got_np, ref_torch, *, atol=ATOL, rtol=RTOL, what=""):
    ref_np = ref_torch.detach().cpu().numpy().astype(np.float64)
    got_np = np.asarray(got_np, dtype=np.float64)
    ok = np.allclose(got_np, ref_np, atol=atol, rtol=rtol)
    if not ok:
        max_abs = float(np.max(np.abs(got_np - ref_np)))
        denom = np.abs(ref_np) + 1e-30
        max_rel = float(np.max(np.abs(got_np - ref_np) / denom))
        raise AssertionError(
            f"{what}: parity FAILED max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"(atol={atol}, rtol={rtol})")
    return float(np.max(np.abs(got_np - ref_np)))


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
#  Per-optimizer parity tests.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_adamw_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.05
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)   # nonzero state
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    # Several steps with state threaded forward.
    for t in range(1, 5):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv = _run_interp(
            launch_adamw_step,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(g)),
            dict(lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, bc1=bc1, bc2=bc2),
            n_out=3)
        rp, rm, rv = ref_adamw_step(_t(p), _t(g), _t(m), _t(v),
                                    lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, t=t)
        _close(gp, rp, what=f"adamw p (t={t})")
        _close(gm, rm, what=f"adamw m (t={t})")
        _close(gv, rv, what=f"adamw v (t={t})")
        p, m, v = gp.astype(np.float32), gm.astype(np.float32), gv.astype(np.float32)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_lion_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, wd = 1e-3, 0.9, 0.99, 0.05
    p = r.standard_normal(size).astype(np.float32)
    ea = (r.standard_normal(size) * 0.2).astype(np.float32)
    for t in range(1, 5):
        g = r.standard_normal(size).astype(np.float32)
        gp, gea = _run_interp(
            launch_lion_step,
            (jnp.asarray(p), jnp.asarray(ea), jnp.asarray(g)),
            dict(lr=lr, beta1=b1, beta2=b2, wd=wd),
            n_out=2)
        rp, rea = ref_lion_step(_t(p), _t(g), _t(ea), lr=lr, beta1=b1, beta2=b2, wd=wd)
        # Lion's sign() at exactly-0 interp differs (kernel uses jnp.sign(0)=0,
        # ref guards sign(0)=0 too) — random grads make interp==0 ~never.
        _close(gp, rp, what=f"lion p (t={t})")
        _close(gea, rea, what=f"lion ea (t={t})")
        p, ea = gp.astype(np.float32), gea.astype(np.float32)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_grokadamw_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, eps, wd = 3e-4, 0.9, 0.999, 1e-8, 0.01
    alpha, lamb = 0.98, 2.0
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    ema = (r.standard_normal(size) * 0.1).astype(np.float32)
    for t in range(1, 5):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv, gema = _run_interp(
            launch_grokadamw_step,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(ema),
             jnp.asarray(g)),
            dict(alpha=alpha, lamb=lamb, lr=lr, beta1=b1, beta2=b2, eps=eps,
                 wd=wd, bc1=bc1, bc2=bc2),
            n_out=4)
        rp, rm, rv, rema = ref_grokadamw_step(
            _t(p), _t(g), _t(m), _t(v), _t(ema),
            alpha=alpha, lamb=lamb, lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, t=t)
        _close(gp, rp, what=f"grokadamw p (t={t})")
        _close(gema, rema, what=f"grokadamw ema (t={t})")
        p, m, v, ema = (gp.astype(np.float32), gm.astype(np.float32),
                        gv.astype(np.float32), gema.astype(np.float32))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_grokfast_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, eps, wd = 3e-4, 0.9, 0.999, 1e-8, 0.01
    alpha, lamb = 0.98, 5.0
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    ema = (r.standard_normal(size) * 0.1).astype(np.float32)
    for t in range(1, 5):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv, gema = _run_interp(
            launch_grokfast_step,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(ema),
             jnp.asarray(g)),
            dict(gf_alpha=alpha, gf_lamb=lamb, lr=lr, beta1=b1, beta2=b2,
                 eps=eps, wd=wd, bc1=bc1, bc2=bc2),
            n_out=4)
        rp, rm, rv, rema = ref_grokfast_step(
            _t(p), _t(g), _t(m), _t(v), _t(ema),
            alpha=alpha, lamb=lamb, lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, t=t)
        _close(gp, rp, what=f"grokfast p (t={t})")
        _close(gema, rema, what=f"grokfast ema (t={t})")
        p, m, v, ema = (gp.astype(np.float32), gm.astype(np.float32),
                        gv.astype(np.float32), gema.astype(np.float32))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_neuralgrok_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.0
    alpha, beta = 0.5, 1.0
    H = 16   # deployed psi width (opt_components.cuh kPsiHidden=16)
    # psi_forward: ag[:,None] @ W1[None,:] -> W1 is [H]; h @ W2[:,None] -> W2 is [H].
    W1 = (r.standard_normal(H) * 0.5).astype(np.float32)
    b1 = (r.standard_normal(H) * 0.1).astype(np.float32)
    W2 = (r.standard_normal(H) * 0.5).astype(np.float32)
    b2 = float(r.standard_normal())
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    beta1, beta2 = 0.9, 0.999
    for t in range(1, 4):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - beta1 ** t, 1.0 - beta2 ** t
        gp, gm, gv = _run_interp(
            launch_neuralgrok_step,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(g),
             jnp.asarray(W1), jnp.asarray(b1), jnp.asarray(W2)),
            dict(psi_b2=b2, alpha=alpha, beta=beta, lr=lr, beta1=beta1, beta2=beta2,
                 eps=eps, wd=wd, bc1=bc1, bc2=bc2),
            n_out=3)
        # Reference: psi = 2-layer ReLU MLP on |g|, then g_amp=(psi*alpha+beta)*g.
        psi = ref_neuralgrok_psi_forward(_t(g).abs(), _t(W1), _t(b1), _t(W2), b2)
        rp, rm, rv = ref_neuralgrok_step(
            _t(p), _t(g), _t(m), _t(v), psi_scale=psi,
            alpha=alpha, beta=beta, lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd, t=t)
        _close(gp, rp, what=f"neuralgrok p (t={t})")
        p, m, v = gp.astype(np.float32), gm.astype(np.float32), gv.astype(np.float32)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_looksam_apply_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    alpha = 0.7
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    sam = (r.standard_normal(size) * 0.3).astype(np.float32)
    for t in range(1, 4):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv = _run_interp(
            launch_looksam_apply,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(sam),
             jnp.asarray(g)),
            dict(alpha=alpha, lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd,
                 bc1=bc1, bc2=bc2),
            n_out=3)
        rp, rm, rv = ref_looksam_apply_step(
            _t(p), _t(g), _t(m), _t(v), _t(sam),
            alpha=alpha, lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, t=t)
        _close(gp, rp, what=f"looksam p (t={t})")
        _close(gm, rm, what=f"looksam m (t={t})")
        p, m, v = gp.astype(np.float32), gm.astype(np.float32), gv.astype(np.float32)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_looksam_perturb_setdir_interpret_parity(size, seed):
    r = _rng(seed)
    p = r.standard_normal(size).astype(np.float32)
    g = r.standard_normal(size).astype(np.float32)
    scale = 0.05
    # perturb: returns (p + scale*g, backup); kernel takes scale as scalar kwarg.
    gp, gbackup = _run_interp(
        launch_looksam_perturb,
        (jnp.asarray(p), jnp.asarray(g)),
        dict(scale=scale),
        n_out=2)
    rp = ref_looksam_perturb(_t(p), _t(g), scale=scale)
    _close(gp, rp, what="looksam perturb")
    _close(gbackup, _t(p), what="looksam perturb backup")
    # set_direction: grad_sam - grad_orig.
    g_sam = r.standard_normal(size).astype(np.float32)
    gdir, = _run_interp(
        launch_looksam_set_direction,
        (jnp.asarray(g_sam), jnp.asarray(g)),
        dict(),
        n_out=1)
    rdir = ref_looksam_set_direction(_t(g_sam), _t(g))
    _close(gdir, rdir, what="looksam set_direction")


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_prodigy_interpret_parity(size, seed):
    """Prodigy fused-TU contract: degree-2 partials -> d update -> Adam(d*g)."""
    r = _rng(seed)
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.0
    p = r.standard_normal(size).astype(np.float32)
    p_init = (p - r.standard_normal(size) * 0.5).astype(np.float32)  # drifted from init
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    s_track = (r.standard_normal(size) * 0.1).astype(np.float32)
    d_prev = 0.3
    for t in range(1, 4):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv, gs, gd = _run_interp(
            _prodigy_wrapper,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(s_track),
             jnp.asarray(p_init), jnp.asarray(g)),
            dict(d_prev=float(d_prev), beta1=b1, beta2=b2, eps=eps, wd=wd,
                 bc1=bc1, bc2=bc2),
            n_out=5)
        # Reference: compute d from the degree-2 partials, then the apply step.
        r_sum, s_sum = ref_prodigy_partials(_t(p), _t(p_init), _t(g), d_prev=d_prev)
        # launch_prodigy_step's d candidate guard uses |s_sum|+1e-12 (no early
        # return), so mirror that exactly here for an apples-to-apples d.
        d = float(max(d_prev, float(r_sum) / (abs(float(s_sum)) + 1e-12)))
        rp, rm, rv, rs = ref_prodigy_apply_step(
            _t(p), _t(g), _t(m), _t(v), _t(s_track),
            d=d, beta1=b1, beta2=b2, eps=eps, wd=wd, t=t)
        _close(np.array([gd[0]] if np.ndim(gd) else gd), torch.tensor([d]),
               what=f"prodigy d (t={t})")
        _close(gp, rp, what=f"prodigy p (t={t})")
        _close(gs, rs, what=f"prodigy s_track (t={t})")
        p, m, v, s_track = (gp.astype(np.float32), gm.astype(np.float32),
                            gv.astype(np.float32), gs.astype(np.float32))
        d_prev = d


def _prodigy_wrapper(param, exp_avg, exp_avg_sq, s_track, param_init, grad,
                     *, d_prev, beta1, beta2, eps, wd, bc1, bc2):
    """Adapt launch_prodigy_step (returns scalar d) to a whole-array kernel by
    broadcasting d to the param shape so it can be written to an output ref."""
    new_param, m, v, new_s, d = launch_prodigy_step(
        param, exp_avg, exp_avg_sq, s_track, param_init, grad, d_prev,
        beta1, beta2, eps, wd, bc1, bc2)
    d_arr = jnp.broadcast_to(jnp.asarray(d, jnp.float32), param.shape)
    return new_param, m, v, new_s, d_arr


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_supergrok11_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    alpha = 0.5
    H = 32
    # phi_forward: x=[N,2] @ W1.T -> W1 is [H,2]; h @ W2[:,None] -> W2 is [H].
    W1 = (r.standard_normal((H, 2)) * 0.3).astype(np.float32)
    b1v = (r.standard_normal(H) * 0.1).astype(np.float32)
    W2 = (r.standard_normal(H) * 0.3).astype(np.float32)
    b2v = float(r.standard_normal())
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    sharp = (r.standard_normal(size) * 0.5).astype(np.float32)
    momentum = (r.standard_normal(size) * 0.5).astype(np.float32)
    for t in range(1, 4):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv, gmu = _run_interp(
            launch_supergrok11_step,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(momentum),
             jnp.asarray(g), jnp.asarray(sharp), jnp.asarray(momentum),
             jnp.asarray(W1), jnp.asarray(b1v), jnp.asarray(W2)),
            dict(phi_b2=b2v, alpha=alpha, lr=lr, beta1=b1, beta2=b2, eps=eps,
                 wd=wd, bc1=bc1, bc2=bc2),
            n_out=4)
        # Reference: mu = phi(g, sharp); sigmoid-of-cosine gate; sg11 step.
        # ref_sg_phi_forward is the PER-ELEMENT meta-net (scalar grad/sharp); to
        # evaluate it on the whole [N] vector at once, add a trailing axis so the
        # [H] hidden broadcasts over N -> pre is [N, H], output is [N].
        mu = ref_sg_phi_forward(_t(g).unsqueeze(-1), _t(sharp).unsqueeze(-1),
                                _t(W1).reshape(-1), _t(b1v), _t(W2), b2v)
        gn = (_t(g) * _t(momentum)).sum()
        gdg = (_t(g) * _t(g)).sum()
        gdm = (_t(momentum) * _t(momentum)).sum()
        # gate = sigmoid(gate_temp * cos) — the canonical sg11_finalize_gate. The
        # interp call omits gate_temperature, so the Pallas step uses its default 5.0.
        cos = gn / torch.sqrt(gdg * gdm + 1e-12)
        gate = torch.sigmoid(5.0 * cos)
        rp, rm, rv = ref_sg11_step(
            _t(p), _t(g), _t(m), _t(v), mu, gate=float(gate), alpha=alpha,
            lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, t=t)
        _close(gmu, mu, what=f"sg11 mu (t={t})")
        _close(gp, rp, what=f"sg11 p (t={t})")
        p, m, v = gp.astype(np.float32), gm.astype(np.float32), gv.astype(np.float32)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_supergrok15_interpret_parity(size, seed):
    r = _rng(seed)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    alpha_base, alpha_max, gate_global = 0.5, 1.0, 0.8
    H = 32
    W1 = (r.standard_normal((H, 2)) * 0.3).astype(np.float32)
    b1v = (r.standard_normal(H) * 0.1).astype(np.float32)
    W2 = (r.standard_normal(H) * 0.3).astype(np.float32)
    b2v = float(r.standard_normal())
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    sharp = (r.standard_normal(size) * 0.5).astype(np.float32)
    mu_buf = (r.standard_normal(size) * 0.1).astype(np.float32)
    for t in range(1, 4):
        g = r.standard_normal(size).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv, gmu = _run_interp(
            launch_supergrok15_step,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(mu_buf),
             jnp.asarray(g), jnp.asarray(sharp),
             jnp.asarray(W1), jnp.asarray(b1v), jnp.asarray(W2)),
            dict(phi_b2=b2v, gate_global=gate_global, alpha_base=alpha_base,
                 alpha_max=alpha_max, lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd,
                 bc1=bc1, bc2=bc2),
            n_out=4)
        # ref_sg_phi_forward is the PER-ELEMENT meta-net (scalar grad/sharp); to
        # evaluate it on the whole [N] vector at once, add a trailing axis so the
        # [H] hidden broadcasts over N -> pre is [N, H], output is [N].
        mu = ref_sg_phi_forward(_t(g).unsqueeze(-1), _t(sharp).unsqueeze(-1),
                                _t(W1).reshape(-1), _t(b1v), _t(W2), b2v)
        rp, rm, rv = ref_sg15_step(
            _t(p), _t(g), _t(m), _t(v), mu, gate_global=gate_global,
            alpha_base=alpha_base, alpha_max=alpha_max, lr=lr, beta1=b1, beta2=b2,
            eps=eps, wd=wd, t=t)
        _close(gmu, mu, what=f"sg15 mu (t={t})")
        _close(gp, rp, what=f"sg15 p (t={t})")
        p, m, v = gp.astype(np.float32), gm.astype(np.float32), gv.astype(np.float32)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_supergrok2_interpret_parity(size, seed):
    """SG2 fused-TU contract (the restored grokfast tail): expert-EMA + slow-EMA
    + lamb_eff, matching csrc/algorithms/supergrok2.h::sg2_apply_step and
    ref_sg2_apply_step. Drives synthetic expert_out + slow_state (the meta-net
    CSA/HCA/PEER front end is NOT modelled by the fp64 ref — a day-1 item)."""
    r = _rng(seed)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    alpha, gru_decay, lamb_eff = 0.7, 0.85, 1.5
    # Unused-by-math placeholders for the fused-TU signature.
    sharpness = jnp.asarray(r.standard_normal(size).astype(np.float32))
    A_log = jnp.zeros((1,), jnp.float32)
    proj_W = jnp.zeros((1, 1), jnp.float32)
    proj_b = jnp.zeros((1,), jnp.float32)
    p = r.standard_normal(size).astype(np.float32)
    m = (r.standard_normal(size) * 0.1).astype(np.float32)
    v = np.abs(r.standard_normal(size) * 0.1).astype(np.float32)
    mu_state = (r.standard_normal(size) * 0.2).astype(np.float32)
    slow_state = (r.standard_normal(size) * 0.2).astype(np.float32)
    for t in range(1, 5):
        g = r.standard_normal(size).astype(np.float32)
        expert_out = (r.standard_normal(size) * 0.5).astype(np.float32)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        gp, gm, gv, gmu, gslow = _run_interp(
            _sg2_wrapper,
            (jnp.asarray(p), jnp.asarray(m), jnp.asarray(v), jnp.asarray(mu_state),
             jnp.asarray(slow_state), jnp.asarray(g), jnp.asarray(expert_out)),
            dict(sharpness=sharpness, A_log=A_log, proj_W=proj_W, proj_b=proj_b,
                 alpha=alpha, gru_decay=gru_decay, lamb_eff=lamb_eff, lr=lr,
                 beta1=b1, beta2=b2, eps=eps, wd=wd, bc1=bc1, bc2=bc2),
            n_out=5)
        rp, rm, rv, rmu, rslow = ref_sg2_apply_step(
            _t(p), _t(g), _t(m), _t(v), _t(mu_state), _t(slow_state),
            expert_out=_t(expert_out), alpha=alpha, gru_decay=gru_decay,
            lamb_eff=lamb_eff, lr=lr, beta1=b1, beta2=b2, eps=eps, wd=wd, t=t)
        _close(gmu, rmu, what=f"sg2 mu_new (t={t})")
        _close(gslow, rslow, what=f"sg2 slow_new (t={t})")
        _close(gp, rp, what=f"sg2 p (t={t})")
        _close(gm, rm, what=f"sg2 m (t={t})")
        _close(gv, rv, what=f"sg2 v (t={t})")
        p, m, v, mu_state, slow_state = (
            gp.astype(np.float32), gm.astype(np.float32), gv.astype(np.float32),
            gmu.astype(np.float32), gslow.astype(np.float32))


def _sg2_wrapper(param, exp_avg, exp_avg_sq, mu_state, slow_state, grad,
                 expert_out, *, sharpness, A_log, proj_W, proj_b, alpha,
                 gru_decay, lamb_eff, lr, beta1, beta2, eps, wd, bc1, bc2):
    return launch_supergrok2_step(
        param, exp_avg, exp_avg_sq, mu_state, slow_state, grad, expert_out,
        sharpness, A_log, proj_W, proj_b, alpha, gru_decay, lamb_eff, lr,
        beta1, beta2, eps, wd, bc1, bc2)


def test_muon_interpret_parity_note():
    """Muon is the ONE optimizer whose canonical kernel is NOT a pure
    elementwise map: launch_muon_step does a Newton-Schulz orthogonalization
    (X@X.T matmuls, a Frobenius reduction) on a 2D weight. That is *not* a
    per-element Pallas body and cannot be validated by the whole-array
    interpret harness used here; it is exercised as a plain jax.jit function
    instead (the NS recurrence has no TPU-only primitive — it is matmul +
    reduce). See test_muon_jit_parity below for the actual parity check; this
    placeholder documents WHY Muon is not in the pallas_call harness.
    """
    assert True


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", [(4, 6), (8, 8), (7, 5), (16, 32)])
def test_muon_jit_parity(seed, shape):
    """Muon NS path parity (jax.jit, not pallas_call — see the note above)."""
    from csrc.backends.pallas.launch_muon import launch_muon_step
    r = _rng(seed)
    lr, momentum, wd, ns_steps = 1e-3, 0.95, 0.01, 5
    p = r.standard_normal(shape).astype(np.float32)
    buf = (r.standard_normal(shape) * 0.1).astype(np.float32)
    for t in range(1, 3):
        g = r.standard_normal(shape).astype(np.float32)
        new_p, new_buf = launch_muon_step(
            jnp.asarray(p), jnp.asarray(buf), jnp.asarray(g),
            lr=lr, momentum=momentum, wd=wd, ns_steps=ns_steps)
        rp, rbuf = ref_muon_step(_t(p), _t(g), _t(buf),
                                 momentum=momentum, lr=lr, wd=wd, ns_steps=ns_steps)
        # NS iterates an ill-conditioned polynomial; fp32 matmul accumulation
        # vs fp64 widens the band slightly. Keep within a loosened fp32 tol.
        _close(np.asarray(new_p, np.float64), rp, atol=2e-3, rtol=5e-3,
               what=f"muon p (t={t}, shape={shape})")
        _close(np.asarray(new_buf, np.float64), rbuf, what=f"muon buf (t={t})")
        p, buf = np.asarray(new_p, np.float32), np.asarray(new_buf, np.float32)


# ---------------------------------------------------------------------------
#  SuperGrok2 LIVE composition path (sg2_update) — the path that actually ships
#  via OPT_STEPS["supergrok2"], distinct from launch_supergrok2_step (the
#  per-element sg2_apply_step twin tested above, which is NOT yet in OPT_STEPS).
#
#  sg2_update runs the REAL CSA/HCA/PEER meta_net for smart_g, then the restored
#  grokfast tail (slow-EMA + lamb_eff) + layer_beta1 Adam, matching the
#  authoritative eager path supergrok2.py::_single_param_step:2388-2406. We do a
#  DIFFERENTIAL reconstruction (NOT a mock): run the same real meta_net_forward
#  in the reference so smart_g cancels bit-for-bit (fp32), leaving only the
#  grokfast+Adam tail to validate in fp64.
# ---------------------------------------------------------------------------

def _sg2_meta_weights(rng, dm=8):
    """NONZERO randomized meta-weights matching _pallas_fused.py's sg2 state
    builder shapes (lines ~949-979), so smart_g departs from g meaningfully."""
    def R(*shape, s=0.3):
        return jnp.asarray((rng.standard_normal(shape) * s).astype(np.float32))
    arrays = {
        "input_proj_W": R(dm, 2), "input_proj_b": R(dm),
        "csa_q_W": R(dm, dm), "csa_k_W": R(dm, dm),
        "csa_v_W": R(dm, dm), "csa_out_W": R(dm, dm),
        "csa_compress_w": R(8),
        "csa_idx_DQ": R(dm, 4), "csa_idx_UQ": R(4, dm), "csa_idx_K": R(dm, 4),
        "hca_q_W": R(dm, dm), "hca_k_W": R(dm, dm),
        "hca_v_W": R(dm, dm), "hca_out_W": R(dm, dm),
        "gru_W_z": R(4, 2 + 2 * dm + 4), "gru_W_r": R(4, 2 + 2 * dm + 4),
        "gru_W_h": R(4, 2 + 2 * dm + 4),
        "gru_b_z": R(4), "gru_b_r": R(4), "gru_b_h": R(4),
        "peer_query_Ws": R(2, dm, 4 + 2 * dm + 2),
        "product_keys_A": R(2, 4, dm // 2), "product_keys_B": R(2, 4, dm // 2),
        "expert_W1": R(16, 4, 1), "expert_b1": R(16, 4),
        "expert_W2": R(16, 1, 4), "expert_b2": R(16, 1),
    }
    structural = {
        "rescale": 1.0, "d_model": dm, "pk_dim": 4, "num_peer_heads": 2,
        "n_heads": 2, "csa_compress": 4, "csa_window": 8, "csa_topk": 4,
        "hca_compress": 4, "indexer_rank": 4, "gru_hidden": 4,
    }
    return arrays, structural


@pytest.mark.parametrize("size", [8, 16, 32])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gate_signal", [0.0, 0.6])
def test_supergrok2_live_update_parity(size, seed, gate_signal):
    """LIVE sg2_update tail parity (gate=0 catches the base-term half of the
    drift fix: effective_g must be smart_g, NOT the old gc + ...; gate!=0
    catches the restored grokfast slow-EMA term)."""
    from csrc.backends.pallas._pallas_fused import (
        sg2_update, meta_net_forward, _meta_weights_from_dict, MetaNetConfig)
    rng = _rng(seed)
    dm = 8
    arrays, structural = _sg2_meta_weights(rng, dm)
    meta_weights = dict(arrays)
    meta_weights.update(structural)
    layer_beta1, beta2, lr, wd, eps = 0.9, 0.999, 1e-3, 0.01, 1e-8
    lamb, ramp, alpha, grad_clip = 2.0, 1.0, 0.7, 1.0
    hyper = dict(layer_beta1=layer_beta1, beta2=beta2, lr=lr, wd=wd, eps=eps,
                 lamb=lamb, ramp=ramp, gate_signal=gate_signal, alpha=alpha,
                 grad_clip=grad_clip)

    p = (rng.standard_normal(size) * 0.5).astype(np.float32)
    st = {
        "exp_avg": (rng.standard_normal(size) * 0.1).astype(np.float32),
        "exp_avg_sq": np.abs(rng.standard_normal(size) * 0.1).astype(np.float32),
        "mu": (rng.standard_normal(size) * 0.1).astype(np.float32),
        "slow": (rng.standard_normal(size) * 0.1).astype(np.float32),
        "sharpness": (rng.standard_normal(size) * 0.3).astype(np.float32),
        "gru_state": (rng.standard_normal((size, 4)) * 0.1).astype(np.float32),
        "step": 2,
    }
    st = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
          for k, v in st.items()}

    config = MetaNetConfig(
        d_model=dm, n_heads=2, csa_compress=4, csa_window=8, csa_topk=4,
        hca_compress=4, indexer_rank=4, num_peer_heads=2,
        num_experts=16, expert_hidden=4, gru_hidden=4, pk_dim=4, rescale=1.0)
    mnw = _meta_weights_from_dict(meta_weights)

    for _ in range(3):
        g = (rng.standard_normal(size) * 1.0).astype(np.float32)
        new_p, new_st = sg2_update(jnp.asarray(g), jnp.asarray(g), st,
                                   meta_weights, hyper)
        # Differential reference: same clip, same REAL meta_net (cancels in fp32),
        # then the grokfast+Adam tail in fp64.
        g32 = jnp.asarray(g)
        gnorm = jnp.linalg.norm(g32.reshape(-1))
        clip = jnp.minimum(1.0, grad_clip / (gnorm + 1e-12))
        gc = np.asarray(g32 * clip, np.float64)
        smart_g, _gru, _ = meta_net_forward(
            jnp.asarray(g) * clip, st["sharpness"], st["gru_state"], mnw, config)
        smart_g = np.asarray(smart_g, np.float64)
        slow0 = np.asarray(st["slow"], np.float64)
        ea0 = np.asarray(st["exp_avg"], np.float64)
        easq0 = np.asarray(st["exp_avg_sq"], np.float64)
        p0 = np.asarray(g, np.float64)   # params == g in this call (see sg2_update args)
        lamb_eff = lamb * ramp * gate_signal
        new_slow = alpha * slow0 + (1.0 - alpha) * gc
        eff = smart_g + lamb_eff * new_slow
        ea = layer_beta1 * ea0 + (1.0 - layer_beta1) * eff
        easq = beta2 * easq0 + (1.0 - beta2) * eff * eff
        step_i = int(st["step"])
        bc1 = 1.0 - layer_beta1 ** step_i
        bc2 = 1.0 - beta2 ** step_i
        update = (ea / bc1) / (np.sqrt(easq / bc2) + eps)
        exp_p = p0 - lr * (update + wd * p0)

        _close(np.asarray(new_p, np.float64), torch.from_numpy(exp_p),
               what=f"sg2_update live p (gate={gate_signal})")
        _close(np.asarray(new_st["slow"], np.float64), torch.from_numpy(new_slow),
               what=f"sg2_update live slow (gate={gate_signal})")

        # Non-vacuity: at gate!=0 the grokfast term must MOVE the result vs gate=0.
        if gate_signal != 0.0:
            hyper0 = dict(hyper); hyper0["gate_signal"] = 0.0
            p_nogf, _ = sg2_update(jnp.asarray(g), jnp.asarray(g), st,
                                   meta_weights, hyper0)
            assert not np.allclose(np.asarray(new_p), np.asarray(p_nogf),
                                   atol=1e-6, rtol=1e-6), (
                "grokfast slow term is vacuous (gate!=0 == gate=0)")
        st = new_st
