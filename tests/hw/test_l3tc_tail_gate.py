"""tests/hw/test_l3tc_tail_gate.py — per-cell L3-TC conversion gate (owner baseline).

The owner baseline directive's per-cell gates for a newly-converted L3-TC cell:
  (1) megakernel-vs-fp64 single-step   — the in-kernel optimizer TAIL must match a
      reference apply of the SAME canonical update on the SAME TC-reduced grad.
  (2) A/A/A determinism                — the production L3-TC step is bit-identical
      across 3 runs from the same init (fixed tile ownership + ascending-k/t reduce).

RE-ANCHOR (task #10): the eager per-op CUDA kernels are being DELETED, so (1) no longer
runs a fresh eager optimizer + opt.step(); the reference is now the PURE-fp64 canonical
transcription (ref_<opt>_step in tests/hw/test_reference_parity.py) applied PER TENSOR on
the kernel's captured grad + zero-init state. bf16 STAYS the compute precision (the kernel
is unchanged); fp64 is only the ground-truth yardstick.

WHY THIS SHAPE (not a full loss-trajectory-vs-eager oracle): the bf16 wgmma fwd+bwd
and the deterministic grad reduction are SHARED with the already-gated adamw TC cell
(decoder 13/13+5, vit 21/21, mamba 5/5 — grad parity vs the bf16-faithful oracle).
A converted cell changes ONLY the per-element tail (apply_optimizer<Opt>), which is
the 14/0-apply-parity math. So the cell-specific NEW surface is exactly the tail, and
the right gate is "did the kernel's tail apply the canonical update to the reduced
grad correctly" — which is (1). The grad itself is validated by the adamw gates +
the determinism check (2) proves no race was introduced by the tail swap.

The reference apply is the canonical fp64 math (transcribed from csrc/algorithms/<opt>.h
— the SAME single source the kernel's apply_optimizer<Opt> compiles from), run on the grad
the kernel returns (fused_train_step(return_grad=True)). bf16-TC kernel vs fp64 reference →
fp32-reorder tol (1e-4); the SAM 2nd-backward surfaces (sharpness/sam_dir) carry the bf16-
TC-vs-fp64 floor tol (3e-2 / 2.5e-2).

Run:
  PYTHONPATH=. python -m tests.hw.test_l3tc_tail_gate                 # all converted
  PYTHONPATH=. python -m tests.hw.test_l3tc_tail_gate --cell lion/decoder
  PYTHONPATH=. python -m pytest tests/hw/test_l3tc_tail_gate.py -q
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

# ── CROSS-CELL ISOLATION (test-state-leakage fix, #19d) ───────────────────────
# The SuperGrok2 gate's per-op ORACLE (ops.supergrok2_batched_step) and the eager
# per-op neuralgrok_fused_step kernel both perturb a device-global (raw-cudaMalloc /
# __constant__ scratch) that a SUBSEQUENT TC SuperGrok2 launch reads — so when the
# 33-cell suite runs back-to-back in ONE pytest process, a contaminated launch makes
# supergrok2/vit FALSELY fail (sharpness collapses to ~0, _sg2_l3tc_gate.py:154; or
# A/A/A bit-determinism breaks). Verified (#19d): each SG2 cell PASSES in isolation
# but FAILS after a prior SG2 cell (its own oracle leaks) OR a prior neuralgrok cell
# (its per-op kernel leaks). The neuralgrok cells are NOT victims (their gate anchors
# to the canonical neuralgrok.h math and only REPORTS the contamination), and adamw /
# the other clean cells are unaffected — so the minimal, guaranteed-correct isolation
# is to run ONLY the SG2 cells in a FRESH SUBPROCESS (the device-global never crosses
# into them), keeping every other cell in-process for speed. The subprocess runs the
# IDENTICAL gate via the existing `--cell` CLI (sys.exit(0/1) + a "=> PASS" verdict
# line); we assert BOTH the returncode AND the verdict line so a hollow exit-0 can't
# slip through. This is test-side only — NO assertion is weakened, NO kernel touched.
_CONTAMINATION_ISOLATED_OPTS = frozenset({"supergrok2"})


def _g():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import grokking_race_v2 as g
    return g


# Cell spec: model, race optimizer key, and a factory that builds the live optimizer
# with the EXACT production hyperparameters (the SAME ones train_<opt> uses). The factory
# is STILL the production driver — it supplies the kernel's scalars via _opt_scalars_from
# inside fused_train_step (lr/betas/wd/alpha/gamma/rho/grad_clip/…), so the kernel runs the
# real mechanism. The (1a/1b) REFERENCE, however, is now the PURE-fp64 canonical transcription
# (ref_<opt>_step), fed those SAME live scalars + the kernel's captured grad (RE-ANCHOR task
# #10: the eager per-op kernels are deleted). A mechanism the kernel's tail drops (grad_clip,
# per-layer β1, the SAM blend) shows up as a param/state mismatch because the references carry
# it explicitly. State layout in fused_train_step is [m|v|extra].
def _adamw_factory(g, params, c):
    # train_adamw: torch.optim.AdamW(lr, betas=(beta1,beta2), weight_decay, fused=True)
    import torch
    return torch.optim.AdamW(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                             weight_decay=c["weight_decay"])


def _lion_factory(g, params, c):
    # train_lion: Lion(lr=lion_lr, betas=(beta1, 0.99), weight_decay=lion_wd)
    return g.Lion(params, lr=c.get("lion_lr", 3e-4), betas=(c["beta1"], 0.99),
                  weight_decay=c.get("lion_wd", 3.0))


def _grokfast_factory(g, params, c):
    # train_grokfast: Grokfast(lr, betas=(beta1,beta2), weight_decay,
    #                          grokfast_alpha, grokfast_lamb)
    return g.Grokfast(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                      weight_decay=c["weight_decay"],
                      grokfast_alpha=c.get("grokfast_alpha", 0.98),
                      grokfast_lamb=c.get("grokfast_lamb", 2.0))


def _grokadamw_factory(g, params, c):
    # train_grokadamw: GrokAdamW(lr, betas, weight_decay, alpha, gamma, kappa,
    #                            lamb, grad_clip, ...) — has grad_clip + gamma + kappa
    return g.GrokAdamW(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                       weight_decay=c["weight_decay"],
                       alpha=c.get("grokadamw_alpha", 0.98),
                       gamma=c.get("grokadamw_gamma", 0.1),
                       kappa=c.get("grokadamw_kappa", 0.1),
                       lamb=c.get("grokadamw_lamb", 2.0),
                       grad_clip=c.get("grokadamw_grad_clip", 1.0))


def _muon_factory(g, params, c):
    # train_muon: Muon(muon_params=2D, params_1d=1D, lr=muon_lr (0.02),
    # momentum=muon_momentum (0.95), weight_decay). The constructor AUTO-SPLITS a
    # single positional `params` iterable by p.ndim (2D → NS, else → AdamW), exactly
    # matching the kernel's kVitMuon2D routing — so we pass m.parameters() as ONE list.
    return g.Muon(list(params), lr=c.get("muon_lr", 0.02),
                  momentum=c.get("muon_momentum", 0.95),
                  weight_decay=c["weight_decay"])


def _looksam_factory(g, params, c):
    # train_looksam: LookSAM(lr, betas=(beta1,beta2), weight_decay, rho=looksam_rho
    # (0.05), k=looksam_k (5), alpha=looksam_alpha (0.7)). The race uses these exact
    # defaults; the gate drives the SAME optimizer. Its state is exp_avg/exp_avg_sq +
    # sam_direction (the kernel's `extra` slice). k=5 so step 1 is a SAM step
    # (should_sam_step: _global_step%k==0, _global_step starts 0) — the gate runs at
    # step 1, exercising the in-kernel SAM 2nd backward.
    return g.LookSAM(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                     weight_decay=c["weight_decay"],
                     rho=c.get("looksam_rho", 0.05), k=c.get("looksam_k", 5),
                     alpha=c.get("looksam_alpha", 0.7))


def _prodigy_factory(g, params, c):
    # train_prodigy: Prodigy(lr=prodigy_lr (1.0), weight_decay, d0=1e-6, d_coef=1.0,
    # betas=(0.9,0.999)). The race uses the defaults; the gate drives the EXACT same
    # optimizer (its state is exp_avg/exp_avg_sq/s + instance scalars _d_lr/_r_ema/
    # _s_ema). The kernel's `extra` slice == Prodigy's `s` trajectory accumulator.
    return g.Prodigy(params, lr=c.get("prodigy_lr", 1.0),
                     weight_decay=c["weight_decay"],
                     betas=(c["beta1"], c["beta2"]))


def _supergrok11_factory(g, params, c):
    # train_supergrok: SuperGrok11(...). The gate drives the REAL optimizer but with TWO
    # gate-specific overrides that make the SAM 2nd backward + meta-net mu NON-VACUOUS at
    # step 1 (no-hollow-pass): (1) warmup_steps=0 so the ramp factor is 1.0 at step 1 (the
    # default warmup_steps=100 ⇒ ramp=0 ⇒ alpha=0 ⇒ mu contributes nothing — eager-faithful
    # but it would make the apply check blind to mu); (2) a NONZERO meta_net rescale (the
    # SharpnessMetaNet inits rescale=0 ⇒ mu≡0 — that hides the whole mu/sharpness pipeline).
    # We force rescale to a small nonzero value AFTER construction so mu = rescale·phi(g,
    # sharpness) is a real, checkable quantity. meta_hidden_dim=32 (the kernel's compile-time
    # kSgPhiHidden — the known prior 64-bug). lamb=1.0 (the production supergrok_lamb default).
    import torch
    opt = g.SuperGrok11(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                        weight_decay=c["weight_decay"],
                        alpha_init=c.get("supergrok_alpha", 0.98),
                        lamb=c.get("supergrok_lamb", 1.0),
                        gamma=c.get("supergrok_gamma", 0.1),
                        warmup_steps=0, warmup_ramp=1,
                        meta_hidden_dim=32,
                        gate_temperature=c.get("supergrok_gate_temp", 5.0))
    # nonzero rescale so mu != 0 (the meta-net forward is exercised, not a no-op).
    with torch.no_grad():
        opt.meta_net.rescale.fill_(0.1)
    opt.meta_net = opt.meta_net.to(next(iter(opt.param_groups[0]["params"])).device)
    opt._weights_dirty = True   # re-extract the (now nonzero-rescale) weights
    return opt


def _supergrok15_factory(g, params, c):
    # train_supergrok15: SuperGrok15(...). Same gate-specific overrides as SG11 (warmup_steps=0
    # ⇒ ramp=1, nonzero rescale ⇒ mu!=0, meta_hidden_dim=32). SG15's gate is the host scalar
    # sigmoid(gate_scale·(acc - gate_thresh)); at step 1 acc=0 so it is a small positive value.
    import torch
    opt = g.SuperGrok15(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                        weight_decay=c["weight_decay"],
                        alpha_init=c.get("supergrok15_alpha", 0.98),
                        lamb=c.get("supergrok15_lamb", 2.0),
                        gamma=c.get("supergrok15_gamma", 0.1),
                        warmup_steps=0, warmup_ramp=1,
                        meta_hidden_dim=32,
                        sam_rho=c.get("supergrok15_sam_rho", 0.05))
    with torch.no_grad():
        opt.meta_net.rescale.fill_(0.1)
    opt.meta_net = opt.meta_net.to(next(iter(opt.param_groups[0]["params"])).device)
    opt._weights_dirty = True
    return opt


def _neuralgrok_canonical_mv(grad, opt_obj, step, p_before=None):
    """The (m, v, p_after) the NeuralGrok tail MUST produce, by the canonical header math.

    Transcribes csrc/algorithms/neuralgrok.h (the SINGLE source the kernel's
    apply_optimizer<NeuralGrok> calls) — the gate's docstring defines its reference as
    exactly this header math:
        psi   = sum_j psi_W2[j] * relu(psi_W1[j]*|g| + psi_b1[j]) + psi_b2   (kPsiHidden=16)
        g_amp = (psi*alpha + beta) * g
        m     = beta1*m_prev + (1-beta1)*g_amp        (m_prev = 0 at step 1, zero-init cache)
        v     = beta2*v_prev + (1-beta2)*g_amp^2
        update= (m/bc1) / (sqrt(v/bc2) + eps);  p_after = p - lr*(update + wd*p)   (AdamW tail)
    computed in fp64 on the SAME captured TC-reduced grad and the SAME psi weights the
    kernel read (opt_obj.psi_pack). Returns flat fp32 (m, v, p_after) for the whole param
    vector; p_after is None when p_before is None (the (1b)-only call shape).

    RE-ANCHOR (task #10): the eager per-op neuralgrok_fused_step CUDA kernel is being
    DELETED, so BOTH (1a) params and (1b) m/v are now anchored to THIS ground-truth fp64
    transcription of neuralgrok.h — the same single source the kernel's apply_optimizer<
    NeuralGrok> compiles from. The kernel matches this canonical math to ~2.7e-7. (1a) used
    to compare the live eager opt.step(); with the eager path gone, the AdamW tail here IS
    the (1a) reference. bf16 stays the compute precision; fp64 is the yardstick.
    """
    dev = grad.device
    g64 = grad.double()
    # GLOBAL grad-norm clip — neuralgrok.h's tail clips the reduced grad (kernel P2.5,
    # st.clip_coef) BEFORE psi+amp. Apply the SAME clip to the reference so it tracks the
    # kernel when the clip FIRES (global grad-norm > grad_clip, e.g. seed 7); inert
    # (coef=1) when norm<=clip. total_norm = sqrt(Σ g²) over the reduced grad == the
    # kernel's P2.5 norm.
    _gclip = float(opt_obj.param_groups[0].get("grad_clip", 0.0))
    if _gclip > 0.0:
        _tn = g64.norm().item()
        if _tn > _gclip:
            g64 = g64 * (_gclip / (_tn + 1e-6))
    ag = g64.abs()
    pack = opt_obj.psi_pack(device=dev).double()
    H = (pack.numel() - 1) // 3
    W1 = pack[0:H]; b1 = pack[H:2 * H]; W2 = pack[2 * H:3 * H]; b2 = pack[3 * H]
    # psi = sum_j W2[j]*relu(W1[j]*|g| + b1[j]) + b2  (per element)
    hid = torch.relu(ag.unsqueeze(1) * W1.reshape(1, H) + b1.reshape(1, H))   # [N,H]
    psi = (hid * W2.reshape(1, H)).sum(1) + b2                                # [N]
    grp = opt_obj.param_groups[0]
    alpha = float(grp["alpha"]); beta = float(grp["beta"])
    beta1, beta2 = float(grp["betas"][0]), float(grp["betas"][1])
    g_amp = (psi * alpha + beta) * g64
    # step 1 from zero-init moments (the kernel's state cache zero-inits [m|v|extra]).
    m = (1.0 - beta1) * g_amp
    v = (1.0 - beta2) * g_amp * g_amp
    p_after = None
    if p_before is not None:
        # AdamW tail (the (1a) reference). bc1=1-β1, bc2=1-β2 at t=1.
        lr = float(grp.get("lr", 1e-3)); eps = float(grp.get("eps", 1e-8))
        wd = float(grp.get("weight_decay", 0.0))
        bc1 = 1.0 - beta1; bc2 = 1.0 - beta2
        pb = p_before.double().to(dev)
        update = (m / bc1) / (torch.sqrt(v / bc2) + eps)
        p_after = (pb - lr * (update + wd * pb)).float()
    return m.float(), v.float(), p_after


def _neuralgrok_factory(g, params, c):
    # train_neuralgrok: NeuralGrok(lr, betas, weight_decay, alpha=neural_alpha,
    #                              beta=neural_beta, num_layers=neural_layers,
    #                              hidden_dim=neural_hidden, grad_clip=neural_grad_clip).
    # PARITY PINS (the kernel's compile-time psi contract — OPTIMIZER_CONFIGS sets
    # these too): num_layers=2 and hidden_dim=16 == kPsiHidden, so the trained MLP IS
    # the deployed MLP (psi_pack maps it 1:1 into the kernel's `extra` slice). grad_clip
    # is forwarded from the config (default 1.0, == train_neuralgrok); the kernel's P2.5
    # clip applies it, and the canonical fp64 reference (_neuralgrok_canonical_mv) applies
    # the SAME clip — so the parity is real at any seed (inert when ‖g‖₂≤grad_clip).
    params = list(params)
    opt = g.NeuralGrok(params, lr=c["lr"], betas=(c["beta1"], c["beta2"]),
                       weight_decay=c["weight_decay"],
                       alpha=c.get("neural_alpha", 10.0),
                       beta=c.get("neural_beta", 4.0),
                       num_layers=c.get("neural_layers", 2),
                       hidden_dim=c.get("neural_hidden", 16),
                       grad_clip=c.get("neural_grad_clip", 1.0))
    # The amplifier is constructed on CPU in __init__; the race moves it to the
    # param device (grokking_race_v2.py:1094 opt.amplifier=opt.amplifier.to(dev)).
    # Mirror that — without it get_weights()/psi_pack() return CPU tensors, and the
    # kernel reads the psi `extra` slice from the wrong device. This is the production
    # device placement, not a test crutch.
    if params:
        opt.amplifier = opt.amplifier.to(params[0].device)
    return opt


# CONVERTED cells (state-gate clean → production-registered). lion's exp_avg cold-
# starts at ZEROS (lion.py), matching the kernel's zero-init state cache, so its tail
# AND state match the real optimizer exactly. adamw is included as the REGRESSION
# guard for the OptId-generic launcher refactor: the production adamw path now flows
# through the same opt_id-switched launcher (opt_id=0) with the unconditional
# ema/psi-net binding — this re-validates that change against the real AdamW (params
# AND m/v state), which the standalone tc_train_step gates (13/13+5, 21/21) do NOT
# cover (they call the cell's own pybind, not the launcher TU dispatch.cpp uses).
_CELLS = {
    "adamw/decoder": dict(model="decoder", opt="adamw", factory=_adamw_factory),
    "adamw/vit":     dict(model="vit",     opt="adamw", factory=_adamw_factory),
    "lion/decoder":  dict(model="decoder", opt="lion", factory=_lion_factory),
    "lion/vit":      dict(model="vit",     opt="lion", factory=_lion_factory),
    # grokfast (cycle 2): the kernel cold-starts ema=grad at step==1
    # (apply_optimizer<Grokfast> in opt_components.cuh), matching the eager
    # ema=grad0 seed (grokfast.py _group_cache), so the (1b) STATE check now passes
    # the ema slice (was rel 0.98 when the kernel inited ema=0). grokfast has no
    # grad_clip and _opt_scalars_from forwards grokfast_alpha/lamb, so the live
    # optimizer's full mechanism reaches the tail.
    "grokfast/decoder": dict(model="decoder", opt="grokfast", factory=_grokfast_factory),
    "grokfast/vit":     dict(model="vit",     opt="grokfast", factory=_grokfast_factory),
    # mamba (cycle-2 directive (c)): the mamba TC kernel is now wired in-_ops, so the
    # same OptId-generic tail runs over the mamba TC-reduced grad. adamw is the
    # regression guard for mamba's wgmma launcher (the scalar mamba×adamw path stays
    # too); lion/grokfast are the new tails. mamba's wgmma kernel needs B%16==0 —
    # fused_train_step truncates the batch (kSeq=8). The cold-start fix applies to
    # mamba's P3 identically (apply_optimizer<Grokfast> is model-independent).
    "adamw/mamba":    dict(model="mamba", opt="adamw", factory=_adamw_factory),
    "lion/mamba":     dict(model="mamba", opt="lion", factory=_lion_factory),
    "grokfast/mamba": dict(model="mamba", opt="grokfast", factory=_grokfast_factory),
    # neuralgrok (decoder + vit): CONVERTED. The amplifier psi-net MLP is in the TC
    # driver (apply_optimizer<NeuralGrok>); the host fills the psi `extra` slice via
    # NeuralGrok.psi_pack() inside fused_train_step, and alpha/beta are forwarded by
    # _opt_scalars_from. RE-ANCHOR (task #10): BOTH (1a) params and (1b) m/v are now vs the
    # CANONICAL fp64 transcription of neuralgrok.h (_neuralgrok_canonical_mv, extended to
    # also return the AdamW-tail p_after) — fed the kernel's OWN psi_pack weights + reduced
    # grad, including the GLOBAL grad-norm clip (kernel P2.5; inert at the default step,
    # fires at e.g. GATE_SEED=7). The eager per-op kernel reference is deleted. mamba×
    # neuralgrok is NOT here (no mamba TC neuralgrok tail).
    "neuralgrok/decoder": dict(model="decoder", opt="neuralgrok",
                               factory=_neuralgrok_factory),
    "neuralgrok/vit":     dict(model="vit", opt="neuralgrok",
                               factory=_neuralgrok_factory),
    # neuralgrok (mamba — wave-2): the mamba TC launcher dispatches opt_id=6
    # (OptId::NeuralGrok) over the mamba TC-reduced grad; the psi pack is scattered
    # into the `extra` slice host-side (model-independent). Same tail math as
    # decoder/vit, so the (1b) reference is canonical neuralgrok.h + the clip-inert
    # guard (run_cell_gate handles both via the opt=="neuralgrok" branches).
    "neuralgrok/mamba":   dict(model="mamba", opt="neuralgrok",
                               factory=_neuralgrok_factory),
    # grokadamw (decoder): CONVERTED. All THREE eager mechanisms land in the bf16 TC
    # path (apply_optimizer<GrokAdamW> + the kernel's P2.5/P3): (i) per-tensor
    # layer-wise β1 = β1·(1-γ)^layer with rebased bc1 (kernel P3, task id t == the
    # flat named_parameters() layer index — this is the mechanism that FAILS the
    # step-1 STATE gate when dropped, m-rel 0.895, and now PASSES); (ii) GLOBAL
    # grad-norm clip via the kernel's deterministic P2.5 reduction (γ/grad_clip
    # thread through FusedScalars); (iii) adaptive-α is a no-op in-context (no losses
    # fed to .step()), so the static α is faithful. NOTE: this single-step gate is
    # NECESSARY but NOT SUFFICIENT — it is BLIND to (ii) (‖g‖₂≈0.72<1 at step 1 ⇒
    # clip inert). Honest registration also rests on the MULTI-STEP parity in
    # _multistep_grokadamw_parity() below (the clip fires by ~step 50). vit×grokadamw
    # stays blocked (this conversion is decoder-only per the cell order).
    "grokadamw/decoder": dict(model="decoder", opt="grokadamw",
                              factory=_grokadamw_factory),
    # grokadamw (vit): CONVERTED (wave-2 vit lane). The SAME 3-mechanism conversion
    # on the vit TC kernel — fused_vit_megakernel_tc now carries the IDENTICAL P2.5
    # global grad-norm clip + P3 per-tensor layer-wise β1 (t == flat kVitOffsets layer
    # index; cls_token is t=0, vs decoder's 30 tensors). Same caveat: this single-step
    # gate is NECESSARY but NOT SUFFICIENT (BLIND to the clip — ‖g‖₂<1 at step 1 ⇒ inert);
    # the load-bearing check is the MULTI-STEP parity (_multistep_grokadamw_parity, run
    # for both decoder + vit). vit×grokadamw was previously in _BLOCKED_EVIDENCE.
    "grokadamw/vit": dict(model="vit", opt="grokadamw",
                          factory=_grokadamw_factory),
    # grokadamw (mamba — wave-2): the mamba TC kernel gained the IDENTICAL P2.5 global
    # grad-norm clip + P3 per-tensor layer-wise β1 as decoder/vit. The mamba launcher
    # routes opt_id=3. Single-step gate is NECESSARY but BLIND to the clip (‖g‖₂ at
    # step 1 < grad_clip ⇒ clip inert); the multi-step parity (clip fires later) is
    # the load-bearing check, mirrored from decoder _multistep_grokadamw_parity.
    "grokadamw/mamba": dict(model="mamba", opt="grokadamw",
                            factory=_grokadamw_factory),
    # prodigy (decoder): CONVERTED (wave-2 decoder lane). STAGED global-d, computed
    # IN-KERNEL (P2.6 phase, between the grad reduction B2 and the optimizer tail P3)
    # — still a SINGLE persistent wgmma launch. The single-step state gate checks the
    # kernel's [m|v|s_track] against the canonical fp64 ref_prodigy_apply_step's
    # m/v/s_track (run_cell_gate's prodigy branch: the third buffer is `s`, not `ema`,
    # d=d0=1e-6 at step 1). At step 1 param_init==params ⇒ r=0 ⇒ d stays at d0=1e-6,
    # so the single-step state gate is NECESSARY but NOT SUFFICIENT — it is BLIND to
    # the d-adaptation (which fires at step≥2). Honest registration rests on the
    # MULTI-STEP parity (_prodigy_multistep_parity: the kernel's d tracks eager; a
    # d-frozen-at-d0 control diverges, proving d-adaptation is load-bearing).
    "prodigy/decoder": dict(model="decoder", opt="prodigy", factory=_prodigy_factory),
    # prodigy (vit): CONVERTED (wave-2 vit lane). The SAME STAGED global-d P2.6 phase
    # on the vit TC kernel. Same caveat: this single-step gate is necessary-not-
    # sufficient (at step 1 param_init==params ⇒ r=0 ⇒ d=d0, so the d-adaptation has
    # not fired yet); the load-bearing check is the MULTI-STEP parity (the d fires by
    # step≥2; _prodigy_multistep_parity --model vit).
    "prodigy/vit": dict(model="vit", opt="prodigy", factory=_prodigy_factory),
    # prodigy (mamba): CONVERTED — the A/A/A race is FIXED. The non-determinism was a
    # register-pressure-induced wgmma-accumulator spill in the mamba TC backward: the
    # selective-scan backward (mb_scan_bwd) cached a large per-thread frame
    # (dB_loc/dC_loc[kSeq][kState] + a_save[kSeq][kState]) that, combined with prodigy's
    # P2.6 d-reduce state, pushed the kernel past the spill threshold and spilled the 64-
    # reg projection-GEMM accumulator in the double-buffer window (the ptxas C7515 hazard)
    # → denormal/NaN grad drift across bit-identical re-runs. FIX (model_stage_mamba3.cuh +
    # model_stage_mamba_tc.cuh): fused per-timestep dB/dC block-reduce (drops the
    # [kSeq][kState] arrays), dropped a_save (recompute adec=exp(dt·A) in the reverse
    # loop, bit-identical), and __noinline__ on mbtc_forward_tile/mbtc_backward_tile so the
    # heavy model frame lives out-of-line → the megakernel's own frame stays small and the
    # GEMM accumulator is register-resident for EVERY instantiation. prodigy/mamba now PASSES
    # A/A/A bit-exact on the production path (loss/grad/param maxd=0 ×4).
    "prodigy/mamba": dict(model="mamba", opt="prodigy", factory=_prodigy_factory),
    # muon (vit): CONVERTED (wave-2 vit lane). STAGED grid-cooperative Newton-Schulz on
    # the 11 2D weights (in-kernel P2.7); 1D weights → AdamW tail. param_tol=2e-3 for
    # the NS 2D path (OPTSTAGES §8); the (1b) STATE check stays 1e-4 (the momentum buf
    # is buf=μ·buf+g, NO NS, so it must match eager exactly — the load-bearing check).
    "muon/vit": dict(model="vit", opt="muon", factory=_muon_factory, param_tol=2e-3),
    # muon (decoder): CONVERTED (wave-2 decoder lane). The SAME STAGED grid-cooperative
    # Newton-Schulz P2.7 phase on the decoder TC kernel — 11 2D weights orthogonalized
    # in-kernel (incl. tok[99,128]/pos[4,128], shapes no vit muon matrix exercised),
    # 1D weights → AdamW aux tail. param_tol=2e-3 for the NS 2D path (OPTSTAGES §8); the
    # (1b) STATE check stays TIGHT (1e-4) — the momentum buf=μ·buf+g has NO NS, so it
    # must match eager exactly (the load-bearing check; the run_cell_gate momentum_buffer
    # branch reads the eager NS-group momentum for the 2D matrices). pos.weight rows=4 is
    # the smallest NS matrix in the whole campaign — this gate exercises it against eager.
    "muon/decoder": dict(model="decoder", opt="muon", factory=_muon_factory, param_tol=2e-3),
    # muon (mamba — wave-2 mamba lane): the SAME STAGED grid-cooperative Newton-Schulz P2.7
    # phase on the mamba TC kernel (fused_mamba_megakernel_tc) — the 13 ndim==2 mamba weights
    # (mbtc::kMbMuon2D: tok[99,128]/pos[8,128]/A_log[256,16]/in_proj[512,128]/x_proj[40,256]/
    # dt_proj[256,8]/out_proj[128,256] × 2 layers + out.weight[97,128]) orthogonalized in-
    # kernel, 1D/non-2D weights (incl. the ndim==3 conv1d.weight) → AdamW aux tail. param_tol=
    # 2e-3 for the NS 2D path (OPTSTAGES §8); the (1b) STATE check stays TIGHT (1e-4) — the
    # momentum buf=μ·buf+g has NO NS, so it must match eager exactly (the run_cell_gate
    # momentum_buffer branch reads the eager NS-group momentum for the 13 2D matrices). CRITICAL:
    # muon/mamba is a SINGLE forward + NS precompute (NOT a 2nd forward), so — UNLIKE the
    # SAM-2nd-pass mamba cells looksam/SG11/15 — it does NOT hit the shared mamba-forward A/A/A
    # race; the (2) A/A/A determinism check below VERIFIES this (it must be bit-exact). A_log
    # [256,16] (rows≫cols) + x_proj [40,256] (cols≫rows) + dt_proj [256,8] exercise NS shapes
    # neither decoder nor vit muon had. fused_train_step truncates B to B%16 (mamba needs it).
    "muon/mamba": dict(model="mamba", opt="muon", factory=_muon_factory, param_tol=2e-3),
    # looksam (decoder — SAM-tier lane): CONVERTED. The MODEL-COUPLED SAM 2nd backward
    # (st.sam_dir = g_sam − g) is an IN-KERNEL phase (P2.4) on the decoder TC kernel: on
    # SAM steps (every k) it perturbs p'=p+(rho/‖g‖)·g, runs a FULL SECOND in-kernel
    # fwd+bwd at p' → g_sam, writes sam_dir=g_sam−g (persisted in the `extra` slice), and
    # restores p; the apply tail blends g_adj=(1−α)g+α·sam_dir → AdamW. The gate runs at
    # step 1 (a SAM step), so the 2nd backward fires. run_cell_gate has a looksam branch
    # (RE-ANCHORED task #10 — the eager per-op path is deleted):
    #   (1a)+(1b) params + m/v — ref_looksam_apply_step fed the kernel's OWN sam_dir
    #        (k_sam) blends the SAME direction the kernel used, then runs the canonical
    #        fp64 AdamW: tight (1e-4) APPLY-TAIL parity (the 14/0 apply math on the kernel's
    #        reduced grad + sam_dir). param_tol stays tight — the apply has no NS-style
    #        accumulation, so a dropped blend / wrong AdamW shows immediately.
    #   (1b-sam) the `extra` slice (== the kernel's sam_dir) is checked against the PURE-fp64
    #        oracle's OWN 2nd backward (the model's fp64 fwd+bwd at the SAME perturbed weights)
    #        → sam_dir_fp = g_sam_fp − g_fp. The kernel's bf16-TC 2nd backward vs this ground-
    #        truth fp64 2nd backward differ only by the bf16-TC floor — the kernel measured
    #        ~1.87e-2 to fp64, so sam_dir is checked at 2.5e-2 (was 2e-2 vs the bf16-faithful
    #        oracle). A REAL parity against ground-truth math, not a self-referential pass; a
    #        kernel that SKIPPED the SAM phase (sam_dir≈0) FAILS this (g_sam_fp−g is O(rho)≫0).
    #        The 2nd backward's correctness is inherited from the shared grad gate, and (2)
    #        A/A/A proves the 2nd pass introduced no race.
    "looksam/decoder": dict(model="decoder", opt="looksam", factory=_looksam_factory,
                            sam_dir_tol=2.5e-2),
    # looksam (vit — SAM-tier lane): CONVERTED. The IDENTICAL in-kernel P2.4 SAM 2nd
    # backward ported onto the ViT TC kernel (fused_vit_megakernel.cuh): on the step-1
    # SAM step it perturbs p'=p+(rho/‖g‖)·g, runs a FULL SECOND ViT fwd+bwd at p' →
    # g_sam (via vittc_forward_tile/vittc_backward_tile + vittc_dw_*/clspos/lnvec into a
    # SEPARATE sam_grad buffer), writes sam_dir=g_sam−g (persisted in `extra`), restores
    # p. (1a/1b) parity uses the canonical fp64 ref_looksam_apply_step + the sam_dir oracle
    # uses the ViT PURE-fp64 oracle (RE-ANCHOR task #10, vit branch in run_cell_gate); (2)
    # A/A/A proves the 2nd pass introduced no race (vit's forward is deterministic — vit
    # Prodigy/Muon are A/A/A-green, same fixed reductions).
    "looksam/vit": dict(model="vit", opt="looksam", factory=_looksam_factory,
                        sam_dir_tol=2.5e-2),
    # supergrok11 (decoder/vit — SAM-tier lane): CONVERTED. The MODEL-COUPLED SAM 2nd backward
    # (P2.4, sharpness=(g_sam−g)²) + the per-TENSOR meta-net mu/gate precompute (P2.45,
    # mu=rescale·phi(g,sharpness), gate=sigmoid(gate_temp·cos(g,mu))) are IN-KERNEL phases; the apply
    # tail (sg11_sweep_b_step: smart_grad=g+(1−gate)·alpha·mu, AdamW) runs in P3. The gate
    # (factory: warmup_steps=0 ⇒ ramp=1, rescale=0.1 ⇒ mu!=0) validates FOUR surfaces vs the
    # canonical fp64 math (run_cell_gate's opt=="supergrok11"/"supergrok15" branch):
    #   (A) sharpness — vs the PURE-fp64 2nd backward (g_sam−g)² (the SAME ground-truth oracle
    #       the looksam sam_dir check uses, squared); a SKIPPED SAM phase (sharpness≈0) fails it.
    #   (B) mu — vs rescale·phi(g, sharpness) (canonical sg11_phi_forward / ref_sg_phi_forward),
    #       on the SAME captured grad + the kernel's OWN sharpness (so the meta-net forward is
    #       validated independent of the 2nd-backward floor).
    #   (1a) params + (1b) m/v — vs ref_sg11_step (smart_grad=g+(1−gate)·alpha·mu, AdamW) with
    #       the kernel's grad/mu/gate, tight 1e-4 (the apply tail math).
    # mamba×supergrok11 is BLOCKED (shared mamba-forward A/A/A race; code-absent in the mamba
    # kernel/launcher). sharpness_tol=3e-2 (RE-ANCHORED task #10: bf16-TC vs PURE-fp64 ground-
    # truth — the kernel measured 2-3× closer to fp64 than the deleted bf16-faithful oracle was,
    # floor ~2.2e-2; was 2e-2 vs bf16-faithful); mu_tol=3e-3 (fp32 phi reorder, SG-family).
    "supergrok11/decoder": dict(model="decoder", opt="supergrok11",
                                factory=_supergrok11_factory,
                                sharpness_tol=3e-2, mu_tol=3e-3),
    "supergrok11/vit": dict(model="vit", opt="supergrok11",
                            factory=_supergrok11_factory,
                            sharpness_tol=3e-2, mu_tol=3e-3),
    # supergrok15 (decoder/vit — SAM-tier lane): CONVERTED. SAME SAM 2nd backward + mu precompute
    # as SG11, but SIMPLER tail: the gate is a HOST SCALAR (sigmoid(accuracy)), NO per-tensor
    # cosine — so the precompute is just mu=rescale·phi(g,sharpness); the apply (sg15_sweep_b_step)
    # does the per-coord alpha clip + smart_grad=g+gate·a·mu + AdamW. Validated vs ref_sg15_step.
    "supergrok15/decoder": dict(model="decoder", opt="supergrok15",
                                factory=_supergrok15_factory,
                                sharpness_tol=3e-2, mu_tol=3e-3),
    "supergrok15/vit": dict(model="vit", opt="supergrok15",
                            factory=_supergrok15_factory,
                            sharpness_tol=3e-2, mu_tol=3e-3),
    # looksam (mamba): CONVERTED — the A/A/A race is FIXED (same root cause + fix as
    # prodigy/mamba above: a register-pressure wgmma-accumulator spill in the mamba TC
    # backward, exposed harder here because the SAM block inlines the heavy fwd+bwd a SECOND
    # time, doubling the frame). The __noinline__ on mbtc_forward_tile/mbtc_backward_tile +
    # the scan-bwd footprint reduction keep the megakernel frame small even with the SAM
    # block + 2nd-pass inline → the GEMM accumulator stays register-resident. The in-kernel
    # P2.4 SAM 2nd backward (fused_mamba_megakernel.cuh + mamba launcher case OptId::LookSAM)
    # now PASSES A/A/A bit-exact on the production path (loss/grad/param maxd=0 ×4), on BOTH
    # a SAM step (sam_dir = g_sam−g recomputed bit-stable) and a SAM-off step. decoder/vit
    # looksam were already A/A/A-clean; mamba now joins them.
    "looksam/mamba": dict(model="mamba", opt="looksam", factory=_looksam_factory,
                          sam_dir_tol=2.5e-2),
    # supergrok11/15 (mamba — wave-4 mamba SG lane): CONVERTED via the FEATURE PORT (the
    # decoder/vit twins ported onto fused_mamba_megakernel.cuh + mega_mamba_real_adamw_tc_
    # launcher.cu): the P2.4 SAM 2nd backward extends to SG (sharpness=(g_sam−g)²) + the
    # per-tensor meta-net mu precompute (P2.45/P3) + the SG11 cosine gate run as in-kernel
    # phases; the launcher gained the OptId::SuperGrok11/15 cases (opt_id 8/9). The shared-
    # mamba-forward A/A/A race is FIXED (commit 0b57f7e — the LookSAM mamba instantiation
    # proved the SAM double-forward is race-free), so the (2) A/A/A determinism check below
    # VERIFIES SG11/15 are bit-exact too. Same 4-surface validation as decoder/vit (sharpness
    # vs the PURE-fp64 2nd backward, mu vs the canonical phi, 1a/1b apply tail). HONEST
    # CAVEAT: SG11/15 mamba reach L3-TC + single-step parity but WON'T GROK on L3 (the meta-net
    # is untrained — a separate owner-approved host-training task, NOT done here).
    "supergrok11/mamba": dict(model="mamba", opt="supergrok11",
                              factory=_supergrok11_factory,
                              sharpness_tol=3e-2, mu_tol=3e-3),
    "supergrok15/mamba": dict(model="mamba", opt="supergrok15",
                              factory=_supergrok15_factory,
                              sharpness_tol=3e-2, mu_tol=3e-3),
    # supergrok2 (decoder/vit — the LAST/hardest cell): CONVERTED via the DEDICATED
    # ops.sg2_fused_step entry. The FULL CSA/HCA/PEER/GRU meta-net AS the optimizer phase
    # (P3-SG2): in-kernel SEGMENTED SORT (STAGE -1, index-tie-break strategy A) → S0..S5,
    # reading st.sharpness from the SAM 2nd backward (P2.4, shared with SG11/15). The SG2
    # gate (tests/hw/_sg2_l3tc_gate.run_sg2_gate) validates FOUR surfaces: (B1) single-step
    # vs the PURE-fp64 oracle_step (RE-ANCHOR task #10; sg2_kernel_mirror's clean numpy fp64
    # reimpl, validated equivalent — the per-op CUDA ORACLE is deleted) max|Δ{param,
    # m,v,mu,slow,gru_state}| < 1e-5; (A/A/A) bit-determinism (the index-tie-break sort is
    # deterministic by construction); (tie-probe) strategy-A (|g|,idx) perm total-order; and
    # the (N>64 CSA fidelity PROBE) which REPORTS the won't-grok divergence (kernel drops
    # idx_UQ, /sqrt(rank) vs /sqrt(d)) — NOT a regression. run_cell_gate short-circuits to it.
    # supergrok2 (mamba — the LAST/hardest cell): PORTED onto the mamba megakernel (P3-SG2:
    # segmented sort STAGE -1 + the sg2_meta_stages CSA/HCA/GRU/PEER pipeline + the SAM 2nd
    # backward) + the DEDICATED mega_mamba_sg2_tc launcher entry. ⚠ A/A/A: the SAM double-
    # forward + segmented sort re-exercise the shared mamba forward (the .sg2_spec.md-flagged
    # risk); the gate is the gate — if A/A/A re-trips, mamba×supergrok2 is landed dormant.
    # Won't GROK (CSA idx_UQ fidelity gap, out of scope).
    "supergrok2/decoder": dict(model="decoder", opt="supergrok2", factory=None),
    "supergrok2/vit": dict(model="vit", opt="supergrok2", factory=None),
    "supergrok2/mamba": dict(model="mamba", opt="supergrok2", factory=None),
}

# BLOCKED cells — kept here (commented) with the state-gate evidence that blocks them,
# so the reason is reproducible. NOT registered in _FUSED_L3_REAL; the gate's
# has_l3_real precondition would (correctly) refuse to run them as "converted".
#  * grokadamw/vit is now CONVERTED (wave-2 vit lane — registered in _CELLS above): the
#    identical P2.5 global-norm clip + P3 per-tensor β1 are wired into the vit TC kernel
#    (fused_vit_megakernel_tc), the vit launcher already routes opt_id=3, and γ/grad_clip
#    thread via FusedScalars. So _BLOCKED_EVIDENCE is now empty for this lane.
_BLOCKED_EVIDENCE = {}


def _build_cell(model, seed=int(os.environ.get("GATE_SEED", "42"))):
    """Build the live race model + one real batch + the L3-REAL flat layout.

    DETERMINISM (load-bearing for gate 2): build_model does NOT reseed from
    c["seed"] internally — it draws from the global RNG — so the A/A/A determinism
    check MUST reseed here, or each rebuild yields different random init weights and
    the gate would compare runs with DIFFERENT inputs (a test bug, not kernel
    non-determinism). We torch.manual_seed(seed) immediately before build_model so
    every _build_cell(model, seed) returns byte-identical initial params."""
    g = _g()
    c = dict(g.DEFAULT_CONFIG)
    c.update({"model_type": model, "use_amp": False, "use_fused": True,
              "compile_model": False, "matmul_precision": "bf16", "seed": seed,
              "frac_train": 0.5, "val_ratio": 0.10, "p": 97})
    dev = torch.device("cuda")
    data = tuple(d.to(dev) for d in g.make_data_for_task(c, seed))
    torch.manual_seed(seed)            # pin build_model's global-RNG draw
    torch.cuda.manual_seed_all(seed)
    m = g.build_model(c, dev)
    return g, c, m, data, dev


def _sg_bf16_sharpness_oracle(model, named_ref, tx, ty, grad, rho, total, dev):
    """The (g_sam − g)² sharpness the SAM 2nd backward MUST produce, via the bf16-FAITHFUL
    fp64 oracle (the SAME instrument the looksam sam_dir check + the first-grad gate use).
    Runs the oracle TWICE — at p and at p'=p+ron·g (ron=rho/‖g‖_global, g the kernel's reduced
    grad) — and returns sharpness_bf = (g_sam_bf − g_bf)² flattened (named layout order). The
    bf16-TC kernel 2nd backward vs this fp64-bf16-faithful 2nd backward differ only by the bf16
    floor (the SAME design floor the looksam sam_dir 2e-2 tol calibrates), so sharpness is
    checked at that floor. A SKIPPED SAM phase (sharpness≈0) fails the liveness assert."""
    if model == "decoder":
        from tests.hw.test_decoder_tc import _bf16_faithful_oracle as _orc
        B = int(tx.shape[0]); B16 = B - (B % 16)
        a0 = tx[:B16].to(torch.long); a1 = ty[:B16].to(torch.long)
        named_d = {n: p.detach() for n, p in named_ref}
        _lo, go = _orc(named_d, a0, a1)
    elif model == "mamba":
        # The mamba twin of the decoder branch: int tokens, the bf16-faithful fp64 mamba
        # oracle (the SAME instrument the looksam/mamba sam_dir check + the first-grad gate
        # use). B%16 truncation matches fused_train_step's wgmma path.
        from tests.hw.test_mamba_tc import _bf16_faithful_mamba_oracle as _orc
        B = int(tx.shape[0]); B16 = B - (B % 16)
        a0 = tx[:B16].to(torch.long); a1 = ty[:B16].to(torch.long)
        named_d = {n: p.detach() for n, p in named_ref}
        _lo, go = _orc(named_d, a0, a1)
    elif model == "vit":
        from tests.hw.test_vit_tc import _bf16_faithful_oracle as _orc
        B = int(tx.shape[0]); B16 = B - (B % 16)
        a0 = tx[:B16].detach().cpu().double(); a1 = ty[:B16].detach().cpu().to(torch.long)
        named_d = {n: p.detach().cpu() for n, p in named_ref}
        _lo, go = _orc(named_d, a0, a1)
    else:
        raise NotImplementedError(f"sg sharpness oracle not wired for {model}")
    gn = torch.sqrt(sum((gv.double() ** 2).sum() for gv in go.values())).item()
    ron = rho / gn
    off = 0; named_pert = {}
    cpu = (model == "vit")
    for n_, p in named_ref:
        k = p.numel()
        base = (p.detach().cpu().double() if cpu else p.detach().double())
        gslice = grad[off:off + k].reshape(p.shape)
        gslice = (gslice.cpu().double() if cpu else gslice.double())
        named_pert[n_] = base + ron * gslice
        off += k
    _lo2, go2 = _orc(named_pert, a0, a1)
    sharp = torch.cat([((go2[n_] - go[n_]).reshape(-1).double()) ** 2
                       for n_, _ in named_ref]).float().to(dev)
    return sharp


def _sg_pure_fp64_sharpness_oracle(model, named_ref, tx, ty, grad, rho, total, dev):
    """The (g_sam − g)² sharpness the SAM 2nd backward MUST produce, via the PURE-fp64
    oracle (the model's fp64 fwd+bwd with NO bf16 storage rounds — the SAME instrument
    scripts/diag_sg_sharpness.py uses, and the ground-truth math the bf16-faithful oracle
    is itself a rounded approximation of). Runs the oracle TWICE — at p and at p'=p+ron·g
    (ron=rho/‖g‖_global, g the kernel's reduced grad) — and returns sharpness_fp =
    (g_sam_fp − g_fp)² flattened (named layout order).

    RE-ANCHOR (task #10): the sharpness / sam_dir checks were against the bf16-faithful
    oracle (a bf16-rounded reference). Now that the eager per-op CUDA kernels are deleted,
    the gate is re-anchored to ground-truth fp64 math: the L3-TC kernel's bf16-TC 2nd
    backward is validated against the PURE-fp64 2nd backward. Measured (diag_sg_sharpness):
    the kernel sits 2-3× CLOSER to fp64 than the bf16-faithful oracle does, so the floor on
    (kernel − pure-fp64) is ~2.2e-2 sharpness / 1.87e-2 sam_dir — the 3e-2 / 2.5e-2 tols
    below are fp64-justified headroom over those measured floors. bf16 STAYS the compute
    precision (the kernel is unchanged); fp64 is only the reference yardstick.

    The pure-fp64 oracle = the model's ``oracle_loss_and_grads`` run on fp64 params
    (decoder_oracle / vit_oracle / mamba_oracle — each computes in the param dtype). For
    mamba this is the published ``_pure_fp64_oracle`` wrapper; decoder/vit have no such
    wrapper, so we call ``oracle_loss_and_grads`` with doubled params directly (identical
    call shape, ``(named, tokens/patches, targets) -> (loss, grads)``)."""
    if model == "decoder":
        from tests.hw.decoder_oracle import oracle_loss_and_grads as _orc
        B = int(tx.shape[0]); B16 = B - (B % 16)
        a0 = tx[:B16].to(torch.long); a1 = ty[:B16].to(torch.long)
        named_d = {n: p.detach().double() for n, p in named_ref}
        _lo, go = _orc(named_d, a0, a1)
    elif model == "mamba":
        # The published pure-fp64 mamba oracle (test_mamba_tc._pure_fp64_oracle doubles
        # the params internally). Same call shape as the bf16-faithful branch above.
        from tests.hw.test_mamba_tc import _pure_fp64_oracle as _orc
        B = int(tx.shape[0]); B16 = B - (B % 16)
        a0 = tx[:B16].to(torch.long); a1 = ty[:B16].to(torch.long)
        named_d = {n: p.detach() for n, p in named_ref}
        _lo, go = _orc(named_d, a0, a1)
    elif model == "vit":
        from tests.hw.vit_oracle import oracle_loss_and_grads as _orc
        B = int(tx.shape[0]); B16 = B - (B % 16)
        a0 = tx[:B16].detach().cpu().double(); a1 = ty[:B16].detach().cpu().to(torch.long)
        named_d = {n: p.detach().cpu().double() for n, p in named_ref}
        _lo, go = _orc(named_d, a0, a1)
    else:
        raise NotImplementedError(f"sg pure-fp64 sharpness oracle not wired for {model}")
    gn = torch.sqrt(sum((gv.double() ** 2).sum() for gv in go.values())).item()
    ron = rho / gn
    off = 0; named_pert = {}
    cpu = (model == "vit")
    for n_, p in named_ref:
        k = p.numel()
        base = (p.detach().cpu().double() if cpu else p.detach().double())
        gslice = grad[off:off + k].reshape(p.shape)
        gslice = (gslice.cpu().double() if cpu else gslice.double())
        named_pert[n_] = base + ron * gslice
        off += k
    _lo2, go2 = _orc(named_pert, a0, a1)
    sharp = torch.cat([((go2[n_] - go[n_]).reshape(-1).double()) ** 2
                       for n_, _ in named_ref]).float().to(dev)
    return sharp


def _run_sg_cell_gate(cell_key, spec, g, c, m, data, dev, canon, opt_obj, grad,
                      loss, p_after, total, named0, cache, verbose,
                      sg11_warmup_state=None):
    """Dedicated SuperGrok11/15 gate: validate the kernel vs the CANONICAL fp64 math its
    apply_optimizer<SG> calls (ref_sg11_step/ref_sg15_step + ref_sg_phi_forward), plus the
    SAM 2nd backward (sharpness) vs the PURE-fp64 oracle (RE-ANCHOR task #10). Returns
    (ok, detail).

    sg11_warmup_state — None for the cold single-step gate (m=v=0 going in, step=1). For
    the warm-up gate (test_sg11_gate_warmup_catches_momentum_bug) it is a tuple
    (m_start, v_start, p_start, t_step): the post-warmup start-of-step state captured
    BEFORE the gated step, so the reference's cos(grad, MOMENTUM) gate + Adam tail use the
    exact same nonzero momentum the kernel's st.exp_avg holds at gate time."""
    import math
    import torch
    from grokking_optimizers.dispatch import _opt_scalars_from, fused_train_step
    from tests.hw.test_reference_parity import (ref_sg11_step, ref_sg15_step,
                                                ref_sg_phi_forward, ref_sg15_alpha_per_coord)
    model, opt = spec["model"], spec["opt"]
    tx, ty = data[0], data[1]

    # Kernel state slices: [m | v | mu | loss | sharpness(total) | phi_pack].
    kstate = cache[canon]["state"]
    k_m = kstate[0:total]; k_v = kstate[total:2 * total]
    k_mu = kstate[2 * total:3 * total]
    k_sharp = kstate[3 * total + 1:4 * total + 1]

    # The SAME scalars the kernel received (alpha/sg_rescale/rho/gate/looksam_sam).
    scalars = _opt_scalars_from(opt_obj, 1)
    alpha = float(scalars.get("alpha", 0.0))
    sg_rescale = float(scalars.get("sg_rescale", 0.0))
    rho = float(scalars.get("rho", 0.05))
    gate_global = float(scalars.get("gate", 0.0))   # SG15 host scalar (SG11: per-tensor cosine)
    gate_temp = float(scalars.get("gate_temp", 5.0)) # SG11 cosine-gate temperature
    sam_on = float(scalars.get("looksam_sam", 0.0))
    beta1, beta2 = float(scalars["beta1"]), float(scalars["beta2"])
    lr, eps, wd = float(scalars["lr"]), float(scalars["eps"]), float(scalars["weight_decay"])
    alpha_max = float(scalars.get("alpha_max", alpha))

    # Per-tensor (name, offset, numel) in the flat layout.
    off = 0; layout = []
    for n_, p in named0:
        layout.append((n_, off, p.numel(), p.shape)); off += p.numel()

    # ── (A) sharpness vs the PURE-fp64 2nd backward (g_sam−g)². Liveness: a real SAM
    # step makes sharpness O((ron·‖∇‖)²) ≫ 0. The factory uses k=… so step 1 IS a SAM step.
    # RE-ANCHOR (task #10): now compared to ground-truth fp64 math (the eager per-op kernels
    # are deleted), not the bf16-faithful oracle. The kernel measured 2-3× closer to fp64
    # than the bf16-faithful oracle was → tol raised 2e-2→3e-2 (floor ~2.2e-2; see _CELLS).
    g_r, c_r, m_ref, data_r, dev_r = _build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    sharp_ref = _sg_pure_fp64_sharpness_oracle(model, named_ref, tx, ty, grad, rho, total, dev)
    assert sam_on != 0.0, f"{cell_key}: looksam_sam==0 — the SAM 2nd backward did not run at step 1"
    assert sharp_ref.abs().max().item() > 1e-8, (
        f"{cell_key}: oracle sharpness is ~0 (the 2nd backward produced no signal) — vacuous gate")
    sh_abs = (k_sharp - sharp_ref).abs().max().item()
    sh_den = sharp_ref.abs().max().item() + 1e-30
    sharp_rel = sh_abs / sh_den
    sharp_tol = spec.get("sharpness_tol", 3e-2)
    sharp_ok = sharp_rel < sharp_tol
    # Liveness on the KERNEL side too: a kernel that skipped the SAM phase would leave
    # sharpness=0 (the zero-init state) → mu=rescale·phi(g,0)≈small, but the apply would
    # still ≈match; the sharpness check above (k_sharp vs the nonzero oracle) catches it.
    assert k_sharp.abs().max().item() > 1e-8, (
        f"{cell_key}: kernel sharpness is ~0 — the in-kernel SAM 2nd backward did not write it")

    # ── (B) mu vs rescale·phi(g, sharpness). Use the kernel's OWN sharpness (so the meta-net
    # forward is validated independent of the bf16 2nd-backward floor) and the SAME phi weights
    # the kernel read (opt_obj.meta_net — the gate scattered THESE into the state). fp64 phi.
    W1, b1, W2, b2, rescale = opt_obj.meta_net.get_weights()
    H = opt_obj.meta_net.hidden_dim
    W1d = W1.reshape(-1).double().to(dev); b1d = b1.reshape(-1).double().to(dev)
    W2d = W2.reshape(-1).double().to(dev); b2d = float(b2.reshape(-1)[0])
    gK = grad.double()                       # the kernel's reduced grad
    shK = k_sharp.double()                   # the kernel's sharpness
    # ref_sg_phi_forward broadcasts grad_val/sharp_val (each [N,1]) against W1[:,0]/[:,1].
    phi = ref_sg_phi_forward(gK.unsqueeze(1), shK.unsqueeze(1), W1d, b1d, W2d, b2d)  # [N]
    mu_ref = (float(rescale) * phi)
    mu_abs = (k_mu.double() - mu_ref).abs().max().item()
    mu_den = mu_ref.abs().max().item() + 1e-30
    mu_rel = mu_abs / mu_den
    mu_tol = spec.get("mu_tol", 3e-3)
    mu_ok = mu_rel < mu_tol
    assert mu_ref.abs().max().item() > 1e-10, (
        f"{cell_key}: oracle mu is ~0 (rescale·phi≈0) — the factory must force a nonzero rescale")

    # ── (1a)+(1b) apply tail: params + m/v vs the canonical fp64 ref (ref_sg{11,15}_step) with
    # the kernel's grad + the kernel's mu + the per-tensor gate. The kernel's apply consumes
    # mu=k_mu (the precompute output) — so the reference uses k_mu too (the apply MATH is what
    # (1a/1b) isolate; (B) already validated mu separately).
    #   * Start-of-step state (m/v going INTO this step): at step 1 the cache is zero-init, so
    #     m_start=v_start=0. Under a WARM-UP (sg11_warmup_state passed in) the m/v slices are the
    #     post-warmup state captured BEFORE this gated step — exactly the kernel's st.exp_avg/
    #     exp_avg_sq at gate time.
    #   * SG11 gate = sigmoid(gate_temp·cos(grad, MOMENTUM)) — the 2nd cosine operand is the
    #     START-OF-STEP momentum (m_start), the canonical sg11_finalize_gate signal, NOT mu.
    p_ref = torch.empty(total, device=dev, dtype=torch.float64)
    m_ref_f = torch.empty(total, device=dev, dtype=torch.float64)
    v_ref_f = torch.empty(total, device=dev, dtype=torch.float64)
    p_before_t = torch.cat([p.data.reshape(-1).double() for _, p in named_ref])
    ws = sg11_warmup_state            # None (cold step 1) or (m_start, v_start, p_start, t_step)
    m_start_all = ws[0] if ws is not None else None
    v_start_all = ws[1] if ws is not None else None
    p_start_all = ws[2] if ws is not None else None
    t_step = ws[3] if ws is not None else 1
    for (n_, o, k, sh) in layout:
        gt = grad[o:o + k].double()
        mut = k_mu[o:o + k].double()
        pt = (p_start_all[o:o + k].double() if p_start_all is not None
              else p_before_t[o:o + k])
        zt = torch.zeros(k, dtype=torch.float64, device=dev)
        m0 = m_start_all[o:o + k].double() if m_start_all is not None else zt
        v0 = v_start_all[o:o + k].double() if v_start_all is not None else zt
        if opt == "supergrok11":
            # per-tensor gate (the kernel's finalizer sg11_finalize_gate):
            # cos = <g,m_start>/sqrt(|g|²|m_start|²+1e-12); gate = sigmoid(gate_temp·cos).
            # The 2nd operand is the START-OF-STEP MOMENTUM (m0), NOT mu — the canonical
            # cos(grad,momentum) signal. At cold step 1 m0=0 ⇒ cos=0 ⇒ gate=0.5.
            num = (gt * m0).sum()
            den = torch.sqrt((gt * gt).sum() * (m0 * m0).sum() + 1e-12)
            cos_t = float(num / den) if den > 0 else 0.0
            gate_t = 1.0 / (1.0 + math.exp(-gate_temp * cos_t))
            pn, mn, vn = ref_sg11_step(pt, gt, m0, v0, mut, gate=gate_t, alpha=alpha,
                                       lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd, t=t_step)
        else:
            pn, mn, vn = ref_sg15_step(pt, gt, m0, v0, mut, gate_global=gate_global,
                                       alpha_base=alpha, alpha_max=alpha_max, lr=lr,
                                       beta1=beta1, beta2=beta2, eps=eps, wd=wd, t=t_step)
        p_ref[o:o + k] = pn; m_ref_f[o:o + k] = mn; v_ref_f[o:o + k] = vn

    def _rel(a, b):
        d = (a.double() - b).abs().max().item(); return d / (b.abs().max().item() + 1e-30)
    param_rel = _rel(p_after, p_ref)
    m_rel = _rel(k_m, m_ref_f)
    v_rel = _rel(k_v, v_ref_f)
    param_tol = spec.get("param_tol", 1e-4)
    param_ok = param_rel < param_tol
    state_ok = (m_rel < 1e-4 and v_rel < 1e-4)
    # WARM-UP MODE: the sharpness (A) PURE-fp64 oracle is built against a FRESH-init
    # model (start-of-run params), but the kernel's sharpness here was produced at the
    # WARMED params — so (A) is not a meaningful comparison under warm-up (the gate's
    # cos(grad,momentum) fix is validated by (1a)/(1b)/(B), which DO use the warmed
    # start-of-step state; B uses the kernel's OWN sharpness). So (A) is informational
    # under warm-up and the (2) A/A/A is owned by the warm-up caller (which re-runs the
    # whole warm-up+gated sequence 3×, not a cold step-1 rebuild).
    warmup_mode = sg11_warmup_state is not None
    if verbose:
        print(f"  (1a) params vs canonical SG: max-rel={param_rel:.3e} (tol {param_tol:.0e}) "
              f"{'OK' if param_ok else 'FAIL'}  loss={loss:.5f}", flush=True)
        print(f"  (1b) STATE vs canonical SG: m={m_rel:.3e} v={v_rel:.3e} (tol 1e-4) "
              f"{'OK' if state_ok else 'FAIL'}", flush=True)
        print(f"  (A) sharpness vs PURE-fp64 2nd backward: rel={sharp_rel:.3e} "
              f"(tol {sharp_tol:.0e}, bf16-TC vs ground-truth fp64 floor) "
              f"{'OK' if sharp_ok else ('INFO(warmup)' if warmup_mode else 'FAIL')}",
              flush=True)
        print(f"  (B) mu vs rescale·phi(g,sharpness): rel={mu_rel:.3e} (tol {mu_tol:.0e}) "
              f"{'OK' if mu_ok else 'FAIL'}  [rescale={float(rescale):.3g} alpha={alpha:.3g} "
              f"gate_global={gate_global:.3g}]", flush=True)
    # The gate-momentum fix is carried by (1a)+(1b)+(B). Under warm-up, (A) is
    # informational (fresh-model oracle vs warmed kernel) and the A/A/A (2) is the
    # caller's; the cold single-step gate keeps the full (A)+(2) teeth.
    tail_ok = param_ok and state_ok and mu_ok and (sharp_ok or warmup_mode)
    if warmup_mode:
        return (tail_ok, dict(param_rel=param_rel, sharp_rel=sharp_rel, mu_rel=mu_rel,
                              m_rel=m_rel, v_rel=v_rel, loss=loss))

    # ── (2) A/A/A determinism: re-run the production step 3× from the SAME init.
    def _one():
        g2, c2, m2, data2, dev2 = _build_cell(model)
        tx2, ty2 = data2[0], data2[1]
        opt2 = spec["factory"](g2, m2.parameters(), c2)
        cc = {}
        L, G = fused_train_step(canon, opt, m2, opt2, tx2, ty2, state_cache=cc,
                                step=1, return_grad=True, gemm_impl="wgmma")
        P = torch.empty(total, device=dev2)
        o = 0
        for _, p in m2.named_parameters():
            if p.requires_grad:
                k = p.numel(); P[o:o + k].copy_(p.data.reshape(-1)); o += k
        # also snapshot the mu + sharpness slices (the new SG machinery) for determinism.
        S = cc[canon]["state"]
        MU = S[2 * total:3 * total].clone(); SH = S[3 * total + 1:4 * total + 1].clone()
        return L, G.clone(), P, MU, SH
    L1, G1, P1, MU1, SH1 = _one(); L2, G2, P2, MU2, SH2 = _one(); L3, G3, P3, MU3, SH3 = _one()
    det_ok = (L1 == L2 == L3 and torch.equal(G1, G2) and torch.equal(G2, G3)
              and torch.equal(P1, P2) and torch.equal(P2, P3)
              and torch.equal(MU1, MU2) and torch.equal(MU2, MU3)
              and torch.equal(SH1, SH2) and torch.equal(SH2, SH3))
    if verbose:
        print(f"  (2) A/A/A determinism: loss {L1:.6f}/{L2:.6f}/{L3:.6f}  "
              f"grad-eq={torch.equal(G1,G2) and torch.equal(G2,G3)}  "
              f"param-eq={torch.equal(P1,P2) and torch.equal(P2,P3)}  "
              f"mu-eq={torch.equal(MU1,MU2) and torch.equal(MU2,MU3)}  "
              f"sharp-eq={torch.equal(SH1,SH2) and torch.equal(SH2,SH3)}  "
              f"{'OK' if det_ok else 'FAIL'}", flush=True)
    ok = tail_ok and det_ok
    return ok, dict(param_rel=param_rel, sharp_rel=sharp_rel, mu_rel=mu_rel,
                    loss=loss, det=det_ok)


def run_sg11_warmup_gate(cell_key, warmup_steps=2, verbose=True):
    """WARM-UP SG11 gate — the divergence-exercising check the cold single-step gate is
    BLIND to (in the worst case mu≈0 ∧ momentum=0 at step 1 ⇒ cos(grad,mu)==cos(grad,
    momentum)==gate(0.5) — the two formulations COINCIDE). After ≥1 warm-up step BOTH
    mu≠0 AND momentum (exp_avg)≠0, so the BUGGY cos(grad,mu) gate and the CANONICAL
    cos(grad,momentum) gate genuinely differ.

    Runs `warmup_steps` real L3-TC steps (shared state_cache so exp_avg accumulates),
    captures the start-of-step m/v/p going INTO the gated step (= the kernel's
    st.exp_avg/exp_avg_sq/params at gate time), then runs the gated step and validates
    the kernel vs the CORRECTED cos(grad,momentum) reference (_run_sg_cell_gate with
    sg11_warmup_state). Returns (ok, detail). detail carries:
      * gate_div — max |gate_momentum − gate_mu| over tensors (>0 proves the warm-up
        exercises the divergence the bug lives in; the OLD kernel computes the gate_mu
        column and must FAIL this corrected reference).
      * mom_norm — the start-of-step momentum L2 norm (must be ≫0 — a live warm-up)."""
    import math
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step,
                                              _opt_scalars_from)
    spec = _CELLS[cell_key]
    model, opt = spec["model"], spec["opt"]
    assert opt == "supergrok11", f"{cell_key}: warmup gate is SG11-only"
    g, c, m, data, dev = _build_cell(model)
    canon = canonicalize_model(model)
    assert has_l3_real(canon, opt), f"{cell_key}: not L3-REAL"
    assert gemm_impl_for_cell(canon, opt, "bf16") == "wgmma", f"{cell_key}: not wgmma"
    tx, ty = data[0], data[1]
    # Build the live optimizer with a LARGE meta-net rescale so the gate-dependent
    # smart_grad term (1−gate)·alpha·mu, mu=rescale·phi, is well ABOVE the param
    # tolerance (1e-4). With the default rescale=0.1 the mu term is too small to move
    # params measurably, so a wrong gate (cos(grad,mu) vs cos(grad,momentum)) hides
    # under the tolerance even when the gates diverge by ~0.8 — the param check would
    # falsely PASS the OLD kernel. A big rescale makes the gate genuinely load-bearing.
    import torch as _t
    opt_obj = spec["factory"](g, m.parameters(), c)
    with _t.no_grad():
        opt_obj.meta_net.rescale.fill_(2.0)
    opt_obj._weights_dirty = True
    named0 = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named0)

    def _flat_params():
        out = torch.empty(total, device=dev); o = 0
        for _, p in named0:
            k = p.numel(); out[o:o + k].copy_(p.data.reshape(-1)); o += k
        return out

    # ── WARM-UP: run `warmup_steps` real steps so exp_avg (momentum) becomes nonzero.
    cache = {}
    for s in range(1, warmup_steps + 1):
        fused_train_step(canon, opt, m, opt_obj, tx, ty, state_cache=cache,
                         step=s, return_grad=False, gemm_impl="wgmma")

    # Start-of-step state going INTO the gated step (= the kernel's st.exp_avg/
    # exp_avg_sq/params at gate time): the cache m/v slices + the live (warmed) params.
    kstate = cache[canon]["state"]
    m_start = kstate[0:total].clone()
    v_start = kstate[total:2 * total].clone()
    p_start = _flat_params()
    mom_norm = float(m_start.norm())
    t_step = warmup_steps + 1

    # ── GATED step (step = warmup_steps+1) on the SAME cache (so it reads the warmed
    # momentum), capturing the TC-reduced grad + post-step params (the kernel's gate
    # now uses cos(grad, m_start)).
    loss, grad = fused_train_step(canon, opt, m, opt_obj, tx, ty, state_cache=cache,
                                  step=t_step, return_grad=True, gemm_impl="wgmma")
    p_after = _flat_params()

    # ── (A)/(B) THE DIRECT GATE SURFACE — recover the kernel's ACTUAL per-tensor gate
    # and compare it to BOTH formulations. This is the load-bearing instrument: the
    # post-Adam params and even the m-state are heavily damped (β1·m0 + 0.1·smart_grad,
    # then v-normalized), so a wrong gate is washed out below the 1e-4 param/state
    # tolerance even when the two gate FORMULATIONS differ by ~0.8. The gate itself is
    # NOT damped, so recovering it exposes the bug cleanly.
    #
    # The kernel wrote k_m = β1·m0 + (1−β1)·smart_grad with smart_grad = g + (1−gate)·
    # alpha·mu and a per-TENSOR scalar gate. So smart_grad_kernel = (k_m − β1·m0)/(1−β1),
    # and (1−gate)·alpha = <smart_grad_kernel − g, mu> / <mu, mu> (least-squares over the
    # tensor since gate is one scalar). gate_kernel = 1 − that/alpha. Compare to:
    #   gate_momentum = sigmoid(gate_temp·cos(grad, m0))   — CANONICAL (FIXED kernel)
    #   gate_mu       = sigmoid(gate_temp·cos(grad, k_mu)) — BUGGY     (OLD kernel)
    scalars = _opt_scalars_from(opt_obj, t_step)
    gate_temp = float(scalars.get("gate_temp", 5.0))
    alpha = float(scalars["alpha"]); beta1 = float(scalars["beta1"])
    k_mu = kstate[2 * total:3 * total]
    k_m = kstate[0:total]
    off = 0; gate_div = 0.0
    err_vs_momentum = 0.0   # |gate_kernel − gate_momentum|  → ~0 for FIXED, large for OLD
    err_vs_mu = 0.0         # |gate_kernel − gate_mu|        → ~0 for OLD,   large for FIXED
    for n_, p in named0:
        k = p.numel()
        gt = grad[off:off + k].double()
        mut = k_mu[off:off + k].double()
        m0 = m_start[off:off + k].double()
        km = k_m[off:off + k].double()
        m0b = m_start[off:off + k].double()
        def _gate(b):
            num = (gt * b).sum()
            den = torch.sqrt((gt * gt).sum() * (b * b).sum() + 1e-12)
            cos = float(num / den) if den > 0 else 0.0
            return 1.0 / (1.0 + math.exp(-gate_temp * cos))
        g_mom = _gate(m0); g_mu = _gate(mut)
        gate_div = max(gate_div, abs(g_mom - g_mu))
        # Recover the kernel's gate from the m-state (per-tensor least-squares).
        sg_k = (km - beta1 * m0b) / (1.0 - beta1)
        denom = float((mut * mut).sum())
        if denom > 1e-20 and alpha > 1e-12:
            one_minus_gate = float(((sg_k - gt) * mut).sum()) / (denom * alpha)
            gate_kernel = 1.0 - one_minus_gate
            err_vs_momentum = max(err_vs_momentum, abs(gate_kernel - g_mom))
            err_vs_mu = max(err_vs_mu, abs(gate_kernel - g_mu))
        off += k

    # ── Validate the FIXED kernel vs the CORRECTED cos(grad,momentum) reference, with
    # the warmed start-of-step state threaded through (m0=m_start, gate uses momentum).
    ok, detail = _run_sg_cell_gate(cell_key, spec, g, c, m, data, dev, canon,
                                   opt_obj, grad, loss, p_after, total, named0, cache,
                                   verbose, sg11_warmup_state=(m_start, v_start, p_start, t_step))
    detail["gate_div"] = gate_div
    detail["mom_norm"] = mom_norm
    detail["gate_err_vs_momentum"] = err_vs_momentum
    detail["gate_err_vs_mu"] = err_vs_mu
    # (A)/(B) VERDICT on the direct gate surface: the kernel's recovered gate must match
    # the CANONICAL cos(grad,momentum) (FIXED) and NOT the buggy cos(grad,mu) (OLD).
    # The recovery sg_k=(k_m−β1·m0)/(1−β1) amplifies the bf16 m-state floor by 1/(1−β1)
    # (≈10×), so the FIXED kernel lands at err_vs_momentum≈1e-4..1e-3 — a few×1e-3 floor.
    # The OLD kernel sits at err_vs_momentum≈0.74..0.92 (and err_vs_mu≈1e-3). The two
    # regimes are separated by ~3 orders of magnitude, so generous bands (1e-2 / 1e-1)
    # give a robust, floor-tolerant verdict with no risk of a flip either way.
    gate_ok = (err_vs_momentum < 1e-2) and (err_vs_mu > 1e-1)

    # ── (D) A/A/A determinism on the WARM-UP path: re-run the full warmup+gated
    # sequence 3× from the SAME init; the gated-step params + mu must be bit-identical.
    def _one_warm():
        g2, c2, m2, data2, dev2 = _build_cell(model)
        tx2, ty2 = data2[0], data2[1]
        opt2 = spec["factory"](g2, m2.parameters(), c2)
        with _t.no_grad():
            opt2.meta_net.rescale.fill_(2.0)
        opt2._weights_dirty = True
        cc = {}
        for s in range(1, warmup_steps + 1):
            fused_train_step(canon, opt, m2, opt2, tx2, ty2, state_cache=cc,
                             step=s, return_grad=False, gemm_impl="wgmma")
        L, G = fused_train_step(canon, opt, m2, opt2, tx2, ty2, state_cache=cc,
                                step=t_step, return_grad=True, gemm_impl="wgmma")
        P = torch.empty(total, device=dev2); o = 0
        for _, p in m2.named_parameters():
            if p.requires_grad:
                k = p.numel(); P[o:o + k].copy_(p.data.reshape(-1)); o += k
        MU = cc[canon]["state"][2 * total:3 * total].clone()
        return L, G.clone(), P, MU
    LA, GA, PA, MA = _one_warm(); LB, GB, PB, MB = _one_warm()
    det_ok = (LA == LB and torch.equal(GA, GB) and torch.equal(PA, PB)
              and torch.equal(MA, MB))
    detail["det"] = det_ok

    if verbose:
        print(f"  (WARMUP) steps={warmup_steps} mom_norm={mom_norm:.4e} "
              f"gate_div(momentum vs mu)={gate_div:.4e} "
              f"{'OK(divergent)' if gate_div > 1e-4 else 'WARN(coincident!)'}", flush=True)
        print(f"  (A/B) recovered-gate: |gate_kernel−gate_momentum|={err_vs_momentum:.4e} "
              f"(want <1e-2 → matches CANONICAL) |gate_kernel−gate_mu|={err_vs_mu:.4e} "
              f"(want >1e-1 → NOT the bug) {'OK(cos grad,momentum)' if gate_ok else 'FAIL'}",
              flush=True)
        print(f"  (D) A/A/A determinism (warmup path): loss {LA:.6f}/{LB:.6f}  "
              f"param-eq={torch.equal(PA, PB)} mu-eq={torch.equal(MA, MB)} "
              f"{'OK' if det_ok else 'FAIL'}", flush=True)
    # The warm-up MUST exercise the divergence (else (A) old-kernel-fails is vacuous).
    assert mom_norm > 1e-6, f"{cell_key}: warm-up momentum ~0 — exp_avg did not accumulate"
    assert gate_div > 1e-4, (
        f"{cell_key}: cos(grad,momentum) and cos(grad,mu) gates COINCIDE "
        f"(div={gate_div:.2e}) — the warm-up does not exercise the bug; the (A) "
        f"old-kernel-fails demonstration would be vacuous. Strengthen the warm-up.")
    detail["gate_ok"] = gate_ok
    return (ok and det_ok and gate_ok), detail


def run_cell_gate(cell_key, verbose=True):
    """Run gates (1)+(2) for one converted cell. Returns (ok, detail)."""
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step,
                                              _opt_scalars_from)
    spec = _CELLS[cell_key]
    model, opt = spec["model"], spec["opt"]
    # ── SuperGrok2 dedicated gate (the LAST/hardest cell). SG2's optimizer phase is the
    # FULL CSA/HCA/PEER/GRU meta-net (in-kernel segmented sort + SAM 2nd backward +
    # sg2_meta_stages), validated against the PURE-fp64 oracle_step (RE-ANCHOR task #10;
    # sg2_kernel_mirror's numpy fp64 reimpl of csa_hca_step_one — NOT the deleted per-op
    # CUDA oracle, NOT the eager opt.step()). tests/hw/_sg2_l3tc_gate owns the (B1)+(A/A/A)+
    # (tie-probe)+(N>64 CSA fidelity probe) verdict; short-circuit run_cell_gate to it.
    if opt == "supergrok2":
        from tests.hw._sg2_l3tc_gate import run_sg2_gate
        return run_sg2_gate(model, verbose=verbose)
    g, c, m, data, dev = _build_cell(model)
    canon = canonicalize_model(model)
    # Precondition: the cell must actually be wired L3-TC (wgmma) — else the gate is
    # meaningless. has_l3_real + gemm_impl_for_cell are the production gates.
    assert has_l3_real(canon, opt), f"{cell_key}: not L3-REAL (wire it first)"
    eng = gemm_impl_for_cell(canon, opt, "bf16")
    assert eng == "wgmma", f"{cell_key}: bf16 engine is {eng!r}, expected wgmma"

    tx, ty = data[0], data[1]
    # A live optimizer carrying the EXACT production hyperparameters (drives the
    # kernel's scalars via _opt_scalars_from inside fused_train_step).
    opt_obj = spec["factory"](g, m.parameters(), c)

    # Snapshot the initial flat params (named_parameters order) BEFORE the step.
    named0 = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named0)
    p_before = torch.empty(total, device=dev)
    off = 0
    for _, p in named0:
        n = p.numel(); p_before[off:off + n].copy_(p.data.reshape(-1)); off += n

    # ── (1) megakernel-vs-REAL-EAGER single-step: run ONE production L3-TC step (the
    # SAME route the race takes), capturing the TC-reduced grad + the post-step
    # params. The state cache starts at zero (m=v=extra=0).
    cache = {}
    loss, grad = fused_train_step(canon, opt, m, opt_obj, tx, ty,
                                  state_cache=cache, step=1, return_grad=True,
                                  gemm_impl="wgmma")
    # params after the kernel step (named order → flat).
    p_after = torch.empty(total, device=dev)
    off = 0
    for _, p in named0:
        n = p.numel(); p_after[off:off + n].copy_(p.data.reshape(-1)); off += n

    # ── SuperGrok11/15 dedicated gate (MODEL-COUPLED SAM 2nd backward + meta-net mu).
    # The SG path validates the kernel against the CANONICAL fp64 math the kernel's
    # apply_optimizer<SG> calls — the same single-source as ref_sg11_step/ref_sg15_step — plus
    # the SAM 2nd backward (sharpness) vs the PURE-fp64 oracle (RE-ANCHOR task #10) and mu vs
    # rescale·phi. Short-circuits run_cell_gate (it owns its own (1a/1b)+(A/B)+(2) verdict).
    if opt in ("supergrok11", "supergrok15"):
        ok, detail = _run_sg_cell_gate(cell_key, spec, g, c, m, data, dev, canon,
                                       opt_obj, grad, loss, p_after, total,
                                       named0, cache, verbose)
        return ok, detail

    # REFERENCE = the CANONICAL fp64 math (the SAME single source — csrc/algorithms/
    # <opt>.h — the kernel's apply_optimizer<Opt> compiles from), applied PER TENSOR on
    # the kernel's OWN captured TC-reduced grad + the kernel's pre-step params + ZERO-init
    # m/v/extra (the state cache zero-inits at step 1), exactly like _run_sg_cell_gate.
    # RE-ANCHOR (task #10): the eager per-op CUDA kernels are being DELETED, so the gate no
    # longer builds a fresh eager optimizer + opt_ref.step(); it drives the ground-truth
    # ref_<opt>_step transcriptions in tests/hw/test_reference_parity.py. bf16 stays the
    # compute precision (the kernel is unchanged); fp64 is only the reference yardstick. A
    # mechanism the kernel's tail drops (grad_clip, per-layer β1, the SAM blend) SHOWS UP
    # here as a param/state mismatch — the references carry those mechanisms explicitly.
    from tests.hw.test_reference_parity import (
        ref_adamw_step, ref_lion_step, ref_grokfast_step, ref_grokadamw_step,
        ref_looksam_apply_step, ref_prodigy_apply_step, ref_muon_step)
    scalars = _opt_scalars_from(opt_obj, 1)
    beta1 = float(scalars["beta1"]); beta2 = float(scalars["beta2"])
    lr = float(scalars["lr"]); eps = float(scalars["eps"])
    wd = float(scalars["weight_decay"])

    # Per-tensor (name, offset, numel, shape, ndim) in the flat layout (named order).
    off = 0; layout = []
    for n_, p in named0:
        layout.append((n_, off, p.numel(), tuple(p.shape), p.ndim)); off += p.numel()

    # GLOBAL grad-norm clip coefficient (kernel P2.5: grokadamw + neuralgrok clip the
    # reduced grad BEFORE the tail; inert coef=1 when ‖g‖₂ ≤ grad_clip). Same guard as
    # _neuralgrok_canonical_mv. At the default gated step it is inert (‖g‖₂<1); it FIRES
    # at e.g. GATE_SEED=7, and the reference then tracks the kernel exactly.
    clip_coef = 1.0
    _gclip = float(opt_obj.param_groups[0].get("grad_clip", 0.0))
    if _gclip > 0.0:
        _tn = grad.double().norm().item()
        if _tn > _gclip:
            clip_coef = _gclip / (_tn + 1e-6)

    # LOOKSAM: the kernel ran the in-kernel SAM 2nd backward (P2.4) on this (step-1) SAM
    # step, writing sam_dir = g_sam − g into the `extra` state slice and BLENDING
    # g_adj=(1−α)g+α·sam_dir in the apply (ref_looksam_apply_step). The gate validates two
    # surfaces, BOTH now anchored to ground-truth fp64 (the eager per-op path is deleted):
    #   (A) sam_dir parity — vs the PURE-fp64 oracle's OWN 2nd backward, perturbed by ron·g
    #       (ron=rho/‖g‖_global, g the kernel's reduced grad): sam_dir_fp = g_sam_fp − g_fp,
    #       compared at the ground-truth fp64 floor (sam_dir_tol=2.5e-2 — RE-ANCHOR task #10;
    #       was 2e-2 vs the bf16-faithful oracle; the kernel measured ~1.87e-2 to fp64).
    #   (1a)+(1b) apply tail — ref_looksam_apply_step fed the kernel's OWN sam_dir (k_sam)
    #       blends the SAME direction the kernel used, isolating params + m/v to the apply
    #       math at tight tol. A SKIPPED SAM phase (sam_dir≈0) fails (A) — ‖sam_dir_fp‖≫0.
    looksam_sam_dir_rel = None
    k_sam = None
    if opt == "looksam":
        kstate0 = cache[canon]["state"]
        k_sam = kstate0[2 * total:3 * total]           # the kernel's cached sam_dir
        rho = float(scalars["rho"])
        # PURE-fp64 sam_dir oracle (per-model, perturbed by ron·g) — the SAME ground-truth
        # instrument the SG sharpness check uses. _sg_pure_fp64_sharpness_oracle returns
        # (g_sam−g)² (squared); here we need the signed (g_sam−g), so inline the per-model
        # oracle (decoder/vit/mamba) twice (at p and p'=p+ron·g) like the SG helper does.
        g_r, c_r, m_ref, data_r, dev_r = _build_cell(model)
        named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
        if model == "decoder":
            from tests.hw.decoder_oracle import oracle_loss_and_grads as _orc
            B = int(tx.shape[0]); B16 = B - (B % 16)
            a0 = tx[:B16].to(torch.long); a1 = ty[:B16].to(torch.long)
            named_d = {n: p.detach().double() for n, p in named_ref}
            _lo, grads_o = _orc(named_d, a0, a1)
            cpu = False
        elif model == "vit":
            from tests.hw.vit_oracle import oracle_loss_and_grads as _orc
            B = int(tx.shape[0]); B16 = B - (B % 16)
            a0 = tx[:B16].detach().cpu().double(); a1 = ty[:B16].detach().cpu().to(torch.long)
            named_d = {n: p.detach().cpu().double() for n, p in named_ref}
            _lo, grads_o = _orc(named_d, a0, a1)
            cpu = True
        elif model == "mamba":
            from tests.hw.test_mamba_tc import _pure_fp64_oracle as _orc
            B = int(tx.shape[0]); B16 = B - (B % 16)
            a0 = tx[:B16].to(torch.long); a1 = ty[:B16].to(torch.long)
            named_d = {n: p.detach() for n, p in named_ref}
            _lo, grads_o = _orc(named_d, a0, a1)
            cpu = False
        else:
            raise NotImplementedError(
                f"{cell_key}: looksam sam_dir pure-fp64 oracle not wired for {model} "
                f"(only decoder/vit/mamba fp64 oracles are available here).")
        gn = torch.sqrt(sum((gv.double() ** 2).sum() for gv in grads_o.values())).item()
        ron = rho / gn
        off = 0; named_pert = {}
        for n_, p in named_ref:
            k = p.numel()
            base = (p.detach().cpu().double() if cpu else p.detach().double())
            gslice = grad[off:off + k].reshape(p.shape)
            gslice = (gslice.cpu().double() if cpu else gslice.double())
            named_pert[n_] = base + ron * gslice
            off += k
        _lo2, grads_o2 = _orc(named_pert, a0, a1)
        r_sam = torch.cat([(grads_o2[n_] - grads_o[n_]).reshape(-1).double()
                           for n_, _ in named_ref]).float().to(dev)
        sd_abs = (k_sam - r_sam).abs().max().item()
        sd_den = r_sam.abs().max().item() + 1e-30
        looksam_sam_dir_rel = sd_abs / sd_den
        # Liveness: a real SAM step makes sam_dir O(ron·‖∇‖) ≫ 0 — a skipped phase is ~0.
        assert r_sam.abs().max().item() > 1e-6, (
            f"{cell_key}: oracle sam_dir is ~0 (the 2nd backward produced no direction) "
            f"— the gate's sam_dir parity would be vacuous.")

    # Drive the canonical fp64 ref per tensor. p_ref/m_ref/v_ref/extra_ref are flat fp64.
    p_ref = torch.empty(total, device=dev, dtype=torch.float64)
    r_m = torch.zeros(total, device=dev, dtype=torch.float64)
    r_v = torch.zeros(total, device=dev, dtype=torch.float64)
    r_ema = torch.zeros(total, device=dev, dtype=torch.float64)
    have_v = have_ema = False
    # NEURALGROK is per-VECTOR (psi reads the whole flat grad together) — apply its
    # canonical transcription once over the full vector, then slice into the flat refs.
    if opt == "neuralgrok":
        ng_m, ng_v, ng_p = _neuralgrok_canonical_mv(grad, opt_obj, step=1, p_before=p_before)
        r_m = ng_m.double(); r_v = ng_v.double(); p_ref = ng_p.double(); have_v = True
    else:
        # muon's 1D AdamW group uses INDEPENDENT aux hyperparameters (adamw_lr/betas/eps);
        # the 2D NS group uses momentum (==scalars["beta1"]) + lr/wd. ns_steps from the 2D
        # group (default 5). Read both once.
        muon_mom = beta1
        muon_ns = 5
        aux_lr = aux_b1 = aux_b2 = None
        if opt == "muon":
            for pg in opt_obj.param_groups:
                if "momentum" in pg and "betas" not in pg:
                    muon_mom = float(pg["momentum"]); muon_ns = int(pg.get("ns_steps", 5))
                if pg.get("group_type") == "adamw" or "betas" in pg:
                    ab = pg.get("betas", (0.9, 0.98))
                    aux_lr = float(pg.get("lr", 1e-3)); aux_b1 = float(ab[0]); aux_b2 = float(ab[1])
            if aux_lr is None:
                aux_lr, aux_b1, aux_b2 = lr, 0.9, 0.98
        for (n_, o, k, sh, nd) in layout:
            g_t = (grad[o:o + k].double() * clip_coef)   # post-clip (inert when coef=1)
            p_t = p_before[o:o + k].double().to(dev)
            z = torch.zeros(k, dtype=torch.float64, device=dev)
            if opt == "adamw":
                pn, mn, vn = ref_adamw_step(p_t, g_t, z, z, lr=lr, beta1=beta1,
                                            beta2=beta2, eps=eps, wd=wd, t=1)
                r_m[o:o + k] = mn; r_v[o:o + k] = vn; have_v = True
            elif opt == "lion":
                # lion: betas=(beta1, 0.99); ea cold-starts at 0; m-slice == exp_avg(ea).
                lb2 = float(opt_obj.param_groups[0]["betas"][1])
                pn, ean = ref_lion_step(p_t, g_t, z, lr=lr, beta1=beta1, beta2=lb2, wd=wd)
                r_m[o:o + k] = ean   # no v for lion (the kernel's v-slice is unused)
            elif opt == "grokfast":
                # grokfast.h cold-starts ema=grad at step 1 (grokfast.py _group_cache);
                # seed the ref ema=g_t ⇒ ema_new = α·g+(1−α)·g = g, g_amp=g+lamb·g.
                a = float(scalars["alpha"]); lamb = float(scalars["lamb"])
                pn, mn, vn, eman = ref_grokfast_step(p_t, g_t, z, z, g_t.clone(),
                                                     alpha=a, lamb=lamb, lr=lr,
                                                     beta1=beta1, beta2=beta2, eps=eps,
                                                     wd=wd, t=1)
                r_m[o:o + k] = mn; r_v[o:o + k] = vn; r_ema[o:o + k] = eman
                have_v = True; have_ema = True
            elif opt == "grokadamw":
                # per-tensor β1 = β1·(1−γ)^layeridx (kernel P3; t == flat named layer index).
                # COLD-START ema seed = the UNCLIPPED grad at step 1. grokadamw.py
                # _group_cache clones state["ema"]=p.grad.clone() BEFORE the in-place
                # global clip runs, so the eager (and kernel: opt_components.cuh
                # `st.ema[idx]=grad[idx]` raw, line ~451) seed the RAW grad, while the
                # EMA-filter INPUT + amplification use the CLIPPED grad (g_t). When the
                # clip is inert (coef=1, seed 42) g_t==raw so this is unchanged; when it
                # FIRES (seed 7, global ‖g‖>grad_clip) the ema seed MUST be the unclipped
                # grad or the m/v/ema state diverges from the kernel (params are
                # clip-invariant at step 1, so only the STATE check catches this).
                gamma = float(scalars.get("gamma", 0.0))
                a = float(scalars["alpha"]); lamb = float(scalars["lamb"])
                li = len([1 for (_nn, _oo, _kk, _ss, _dd) in layout
                          if _oo < o])   # this tensor's flat layer index
                b1_i = beta1 * ((1.0 - gamma) ** li)
                g_raw_t = grad[o:o + k].double().to(dev)   # UNCLIPPED grad (ema seed)
                pn, mn, vn, eman = ref_grokadamw_step(p_t, g_t, z, z, g_raw_t,
                                                      alpha=a, lamb=lamb, lr=lr,
                                                      beta1=b1_i, beta2=beta2, eps=eps,
                                                      wd=wd, t=1)
                r_m[o:o + k] = mn; r_v[o:o + k] = vn; r_ema[o:o + k] = eman
                have_v = True; have_ema = True
            elif opt == "looksam":
                # blend the kernel's OWN sam_dir (k_sam); ref_looksam_apply_step does
                # g_adj=(1−α)g+α·sam_dir then AdamW. betas=(beta1, beta2) from the group.
                a = float(scalars["alpha"])
                sam_t = k_sam[o:o + k].double()
                pn, mn, vn = ref_looksam_apply_step(p_t, g_t, z, z, sam_t, alpha=a, lr=lr,
                                                    beta1=beta1, beta2=beta2, eps=eps,
                                                    wd=wd, t=1)
                r_m[o:o + k] = mn; r_v[o:o + k] = vn; have_v = True
                # the `extra` slice (sam_dir) is validated separately (looksam_sam_dir_rel).
            elif opt == "prodigy":
                # d = d0 = 1e-6 at step 1 (param_init==params ⇒ r=0 ⇒ d stays d0). The
                # third state slice is `s` (s_track += d·g). step size is d (not lr).
                d0 = float(scalars.get("d0", 1e-6))
                pn, mn, vn, sn = ref_prodigy_apply_step(p_t, g_t, z, z, z, d=d0,
                                                        beta1=beta1, beta2=beta2, eps=eps,
                                                        wd=wd, t=1)
                r_m[o:o + k] = mn; r_v[o:o + k] = vn; r_ema[o:o + k] = sn
                have_v = True; have_ema = True
            elif opt == "muon":
                if nd == 2:
                    # 2D → Newton-Schulz; the m-slice holds buf=μ·buf+g (NO NS); v-slice 0.
                    pn, bufn = ref_muon_step(p_t.reshape(sh), g_t.reshape(sh), z.reshape(sh),
                                             momentum=muon_mom, lr=lr, wd=wd, ns_steps=muon_ns)
                    pn = pn.reshape(-1); r_m[o:o + k] = bufn.reshape(-1)
                    have_v = True   # 1D tensors below fill v; 2D v-ref stays 0 (kernel writes 0)
                else:
                    # 1D → AdamW with the aux group's hyperparameters.
                    pn, mn, vn = ref_adamw_step(p_t, g_t, z, z, lr=aux_lr, beta1=aux_b1,
                                                beta2=aux_b2, eps=eps, wd=wd, t=1)
                    r_m[o:o + k] = mn; r_v[o:o + k] = vn; have_v = True
            else:
                raise NotImplementedError(
                    f"{cell_key}: no canonical fp64 reference wired for opt={opt!r}")
            p_ref[o:o + k] = pn

    # The kernel and the canonical reference consumed the SAME grad; the only legitimate
    # difference is fp32-vs-fp64 rounding order in the elementwise tail. A DROPPED mechanism
    # (grad_clip scaling every grad, a wrong per-layer β1) makes this large.
    abs_err = (p_after.double() - p_ref).abs().max().item()
    denom = p_ref.abs().max().item() + 1e-30
    rel_err = abs_err / denom
    # Per-cell PARAM tolerance. Default 1e-4 (fp32 reorder; a dropped mechanism blows past
    # it). MUON's 2D-weight params go through a 5-iteration Newton-Schulz whose kernel-fp32
    # vs fp64 NS accumulate to ~2e-3 rel (INTEGRATION-OPTSTAGES §8) — so muon uses 2e-3 for
    # (1a). The (1b) STATE check stays TIGHT (1e-4): the momentum buffer (m) is buf=μ·buf+g,
    # NO NS, so it must match the canonical math exactly (the load-bearing check).
    param_tol = spec.get("param_tol", 1e-4)
    param_ok = rel_err < param_tol
    if verbose:
        print(f"  (1a) params vs canonical fp64: max|Δp|={abs_err:.3e} rel={rel_err:.3e} "
              f"(tol {param_tol:.0e}) {'OK' if param_ok else 'FAIL'}  loss={loss:.5f}", flush=True)

    # ── (1b) STATE vs the canonical fp64 ref — THE DECISIVE CHECK. At step 1 every Adam-
    # family tail collapses to ≈sign(g_amp) in the PARAM update (m/bc1=g_amp, sqrt(v/bc2)=
    # |g_amp|), so (1a) is BLIND to per-layer β1 (γ), amplification magnitude, and STATE
    # INITIALIZATION — all of which live in the optimizer state, not the sign. So we compare
    # the kernel's post-step state slices [m|v|extra] against the canonical references built
    # above. A state mismatch == the kernel is NOT running the canonical algorithm (e.g.
    # grokfast's ema cold-start: header seeds ema=g0, a kernel that inited ema=0 → the ema
    # slice diverges ~50× even when params match). This is what makes (1b) load-bearing.
    kstate = cache[canon]["state"]              # flat [m|v|extra] (3*total)+loss
    k_m = kstate[0:total]; k_v = kstate[total:2 * total]; k_ema = kstate[2 * total:3 * total]
    def _rel(a, b):
        d = (a.double() - b).abs().max().item(); return d / (b.abs().max().item() + 1e-30)
    m_rel = _rel(k_m, r_m)
    v_rel = _rel(k_v, r_v) if have_v else 0.0
    ema_rel = _rel(k_ema, r_ema) if have_ema else None
    # The kernel and the canonical ref consumed the SAME grad; their state must match to
    # fp32-vs-fp64 reorder tol. ema/s only checked when the optimizer HAS that buffer.
    state_ok = (m_rel < 1e-4 and v_rel < 1e-4
                and (ema_rel is None or ema_rel < 1e-4))
    # LOOKSAM: the `extra` slice is the SAM direction (sam_dir=g_sam−g), NOT an EMA/s — so
    # it is NOT in the m/v/ema state check above. It is validated SEPARATELY against the
    # PURE-fp64 2nd backward (looksam_sam_dir_rel, computed above) at the ground-truth fp64
    # floor (sam_dir_tol). Load-bearing for the SAM machinery — fold it into state_ok so a
    # wrong/skipped sam_dir FAILS the gate.
    sam_ok = True
    if opt == "looksam":
        sam_tol = spec.get("sam_dir_tol", 2.5e-2)
        sam_ok = (looksam_sam_dir_rel is not None and looksam_sam_dir_rel < sam_tol)
        state_ok = state_ok and sam_ok
    if verbose:
        ema_s = f"ema={ema_rel:.3e}" if ema_rel is not None else "ema=n/a"
        print(f"  (1b) STATE vs canonical fp64: m={m_rel:.3e} v={v_rel:.3e} {ema_s} "
              f"(tol 1e-4) {'OK' if state_ok else 'FAIL — kernel state != reference'}",
              flush=True)
        if opt == "looksam":
            print(f"  (1b-sam) sam_dir vs PURE-fp64 2nd backward: rel={looksam_sam_dir_rel:.3e} "
                  f"(tol {spec.get('sam_dir_tol', 2.5e-2):.0e}, kernel bf16-TC vs ground-truth "
                  f"fp64 oracle floor) "
                  f"{'OK' if sam_ok else 'FAIL — kernel sam_dir != PURE-fp64 2nd backward'}",
                  flush=True)
    tail_ok = param_ok and state_ok

    # ── (2) A/A/A determinism: re-run the production step 3× from the SAME init,
    # fresh state each time → loss + grad + params must be BIT-identical.
    def _one():
        g2, c2, m2, data2, dev2 = _build_cell(model)
        tx2, ty2 = data2[0], data2[1]
        opt2 = spec["factory"](g2, m2.parameters(), c2)
        cc = {}
        L, G = fused_train_step(canon, opt, m2, opt2, tx2, ty2, state_cache=cc,
                                step=1, return_grad=True, gemm_impl="wgmma")
        P = torch.empty(total, device=dev2)
        o = 0
        for _, p in m2.named_parameters():
            if p.requires_grad:
                k = p.numel(); P[o:o + k].copy_(p.data.reshape(-1)); o += k
        return L, G.clone(), P
    L1, G1, P1 = _one(); L2, G2, P2 = _one(); L3, G3, P3 = _one()
    det_ok = (L1 == L2 == L3 and torch.equal(G1, G2) and torch.equal(G2, G3)
              and torch.equal(P1, P2) and torch.equal(P2, P3))
    if verbose:
        print(f"  (2) A/A/A determinism: loss {L1:.6f}/{L2:.6f}/{L3:.6f}  "
              f"grad-eq={torch.equal(G1,G2) and torch.equal(G2,G3)}  "
              f"param-eq={torch.equal(P1,P2) and torch.equal(P2,P3)}  "
              f"{'OK' if det_ok else 'FAIL'}", flush=True)
    ok = tail_ok and det_ok
    return ok, dict(rel_err=rel_err, loss=loss, det=det_ok)


def _gpu_ready():
    try:
        from grokking_optimizers.dispatch import get_ops
        return torch.cuda.is_available() and hasattr(get_ops(), "fused_step")
    except Exception:
        return False


# ── pytest entry (one test per converted cell; skips cleanly off-GPU) ──
import pytest  # noqa: E402


def _run_cell_gate_subprocess(cell):
    """Run ONE cell's gate in a FRESH python subprocess via the existing `--cell` CLI,
    so a device-global perturbed by a per-op oracle/kernel in this (parent) process can
    NOT contaminate the gated TC launch (the #19d cross-cell test-state-leakage). The
    subprocess executes the IDENTICAL run_cell_gate (no assertion weakened); we assert
    on BOTH its returncode (sys.exit(0) iff the gate passed) AND the captured "=> PASS"
    verdict line (so a non-gate exit-0, e.g. an early SKIP, can never hollow-pass).
    Returns (ok, combined_stdout_stderr) for the caller to assert + surface."""
    env = dict(os.environ)
    # The CLI's `-m tests.hw...` + `import grokking_race_v2` both need ROOT on the path.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(ROOT)])
    proc = subprocess.run(
        [sys.executable, "-m", "tests.hw.test_l3tc_tail_gate", "--cell", cell],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    # A real PASS prints the cell's "=> PASS" line AND "1/1 cells passed" AND exits 0.
    # A SKIP (no GPU/extension) prints "SKIP:" and exits 0 — surface it, don't pass it.
    skipped = "SKIP:" in out and "=> PASS" not in out
    passed = (proc.returncode == 0 and "=> PASS" in out
              and "1/1 cells passed" in out and not skipped)
    return passed, skipped, out


@pytest.mark.hw
@pytest.mark.skipif(not _gpu_ready(), reason="needs built extension on GPU")
@pytest.mark.parametrize("cell", list(_CELLS))
def test_l3tc_cell_gate(cell):
    # CROSS-CELL ISOLATION (#19d): the SG2 cells' per-op oracle (and the eager per-op
    # neuralgrok kernel) leak a device-global into a SUBSEQUENT TC SG2 launch, so the
    # SG2 cells FALSELY fail when the full suite runs back-to-back in one process. Run
    # ONLY those cells in a fresh subprocess (the device-global can't cross in); the
    # clean cells stay in-process for speed. See _CONTAMINATION_ISOLATED_OPTS.
    if _CELLS[cell]["opt"] in _CONTAMINATION_ISOLATED_OPTS:
        passed, skipped, out = _run_cell_gate_subprocess(cell)
        if skipped:
            pytest.skip(f"{cell}: subprocess gate SKIPPED (no GPU/extension)\n{out}")
        assert passed, f"{cell} L3-TC gate failed (isolated subprocess):\n{out}"
        return
    ok, detail = run_cell_gate(cell, verbose=True)
    assert ok, f"{cell} L3-TC gate failed: {detail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="", help="single cell e.g. lion/decoder")
    ap.add_argument("--sg11-warmup", action="store_true",
                    help="run the SG11 WARM-UP gate (cos(grad,momentum) divergence) "
                         "for the supergrok11 cells instead of the cold step-1 gate")
    ap.add_argument("--warmup-steps", type=int, default=2,
                    help="number of warm-up steps before the gated step (default 2)")
    args = ap.parse_args()
    if not _gpu_ready():
        print("SKIP: no built extension / GPU", flush=True)
        return
    if args.sg11_warmup:
        cells = ([args.cell] if args.cell
                 else [k for k in _CELLS if _CELLS[k]["opt"] == "supergrok11"])
        n_ok = 0
        for cell in cells:
            print(f"[l3tc-gate WARMUP] {cell}", flush=True)
            ok, _ = run_sg11_warmup_gate(cell, warmup_steps=args.warmup_steps,
                                         verbose=True)
            print(f"  => {'PASS' if ok else 'FAIL'}\n", flush=True)
            n_ok += int(ok)
        print(f"[l3tc-gate WARMUP] {n_ok}/{len(cells)} SG11 cells passed", flush=True)
        sys.exit(0 if n_ok == len(cells) else 1)
    cells = [args.cell] if args.cell else list(_CELLS)
    n_ok = 0
    for cell in cells:
        print(f"[l3tc-gate] {cell}", flush=True)
        ok, _ = run_cell_gate(cell, verbose=True)
        print(f"  => {'PASS' if ok else 'FAIL'}\n", flush=True)
        n_ok += int(ok)
    print(f"[l3tc-gate] {n_ok}/{len(cells)} cells passed", flush=True)
    sys.exit(0 if n_ok == len(cells) else 1)


if __name__ == "__main__":
    main()
