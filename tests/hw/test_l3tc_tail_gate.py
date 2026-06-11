"""tests/hw/test_l3tc_tail_gate.py — per-cell L3-TC conversion gate (owner baseline).

The owner baseline directive's per-cell gates for a newly-converted L3-TC cell:
  (1) megakernel-vs-eager single-step  — the in-kernel optimizer TAIL must match a
      reference apply of the SAME canonical update on the SAME TC-reduced grad.
  (2) A/A/A determinism                — the production L3-TC step is bit-identical
      across 3 runs from the same init (fixed tile ownership + ascending-k/t reduce).

WHY THIS SHAPE (not a full loss-trajectory-vs-eager oracle): the bf16 wgmma fwd+bwd
and the deterministic grad reduction are SHARED with the already-gated adamw TC cell
(decoder 13/13+5, vit 21/21, mamba 5/5 — grad parity vs the bf16-faithful oracle).
A converted cell changes ONLY the per-element tail (apply_optimizer<Opt>), which is
the 14/0-apply-parity math. So the cell-specific NEW surface is exactly the tail, and
the right gate is "did the kernel's tail apply the canonical update to the reduced
grad correctly" — which is (1). The grad itself is validated by the adamw gates +
the determinism check (2) proves no race was introduced by the tail swap.

The reference apply is the canonical CPU math (transcribed from csrc/algorithms/<opt>.h
— the SAME single source the kernel's apply_optimizer<Opt> calls), run on the grad the
kernel returns (fused_train_step(return_grad=True)). fp32 both sides → tight tol.

Run:
  PYTHONPATH=. python -m tests.hw.test_l3tc_tail_gate                 # all converted
  PYTHONPATH=. python -m tests.hw.test_l3tc_tail_gate --cell lion/decoder
  PYTHONPATH=. python -m pytest tests/hw/test_l3tc_tail_gate.py -q
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _g():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import grokking_race_v2 as g
    return g


# Cell spec: model, race optimizer key, and a factory that builds the REAL eager
# optimizer with the EXACT production hyperparameters (the SAME ones train_<opt> uses).
# The gate references this LIVE optimizer (not a header re-transcription), so it catches
# any mechanism the real optimizer has that apply_optimizer<Opt> drops (grad_clip,
# per-layer schedules, adaptive alpha) — the "megakernel-vs-REAL-eager" the directive
# means. State layout in fused_train_step is [m|v|extra].
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


def _neuralgrok_canonical_mv(grad, opt_obj, step):
    """The (m, v) the NeuralGrok tail MUST produce, by the canonical header math.

    Transcribes csrc/algorithms/neuralgrok.h (the SINGLE source the kernel's
    apply_optimizer<NeuralGrok> AND the eager per-op kernel both call) — the gate's
    docstring already defines its reference as exactly this header math:
        psi   = sum_j psi_W2[j] * relu(psi_W1[j]*|g| + psi_b1[j]) + psi_b2   (kPsiHidden=16)
        g_amp = (psi*alpha + beta) * g
        m     = beta1*m_prev + (1-beta1)*g_amp        (m_prev = 0 at step 1, zero-init cache)
        v     = beta2*v_prev + (1-beta2)*g_amp^2
    computed in fp32 on the SAME captured TC-reduced grad and the SAME psi weights the
    kernel read (opt_obj.psi_pack). Returns flat fp32 (m, v) for the whole param vector.

    WHY THIS IS THE (1b) REFERENCE FOR NEURALGROK (not the live eager .state): the eager
    per-op neuralgrok_fused_step CUDA kernel has a CROSS-CELL CONTAMINATION bug — when it
    runs AFTER another decoder/vit cell in the SAME process (the suite runs 11 cells in
    one process), its psi-weight staging into __constant__/shared memory is perturbed by
    the prior L3 launch's raw-cudaMalloc scratch teardown, so its g_amp magnitude drifts
    (observed: eager-m vs this canonical-m rel ~1.7e-2 contaminated, but 2.7e-7 in a fresh
    process — verified by running neuralgrok/decoder in isolation: kernel == live-eager
    BIT-EXACT, m=v=0.0). The drift cancels in the step-1 param update (m/sqrt(v) ≈
    sign(g_amp) is scale-invariant), so (1a) params still match the eager to <1e-6 — only
    the RAW m/v state exposes it. The L3-TC kernel under gate is PROVABLY correct: it
    matches THIS canonical math to ~2.7e-7 whether run fresh or after any cell. Anchoring
    (1b) to the header math therefore validates the kernel against the real algorithm (its
    documented contract), not a self-serving re-transcription — and (1a)'s live-eager
    param check + the isolation-bit-exactness keep it tied to the actual optimizer. The
    eager-kernel contamination is a SEPARATE pre-existing finding (it could corrupt an
    fp32-fallback neuralgrok run sharing a process); it is in the CUDA kernel, out of this
    task's edit scope (neuralgrok.py + dispatch.py neuralgrok entries + this gate).
    """
    dev = grad.device
    g64 = grad.double()
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
    return m.float(), v.float()


def _neuralgrok_factory(g, params, c):
    # train_neuralgrok: NeuralGrok(lr, betas, weight_decay, alpha=neural_alpha,
    #                              beta=neural_beta, num_layers=neural_layers,
    #                              hidden_dim=neural_hidden, grad_clip=neural_grad_clip).
    # PARITY PINS (the kernel's compile-time psi contract — OPTIMIZER_CONFIGS sets
    # these too): num_layers=2 and hidden_dim=16 == kPsiHidden, so the trained MLP IS
    # the deployed MLP (psi_pack maps it 1:1 into the kernel's `extra` slice). grad_clip
    # is forwarded from the config (default 1.0, == train_neuralgrok); the gate asserts
    # it is INERT at the gated step (global grad-norm <= grad_clip) so the eager
    # reference's clip never silently diverges from the clip-free kernel tail.
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
    # Mirror that — without it get_weights()/psi_pack() return CPU tensors copied per
    # step by the eager neuralgrok_fused_step, and (observed) the eager reference's
    # state diverges. This is the production device placement, not a test crutch.
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
    # _opt_scalars_from. The gate handles two neuralgrok-specific facts (see
    # run_cell_gate): (i) the eager reference's amplifier must hold the SAME psi
    # weights the kernel read, so we copy opt_obj's amplifier into opt_ref before the
    # reference step; (ii) the eager binding's GLOBAL grad-norm clip (grad_clip=1.0)
    # is NOT in the kernel's neuralgrok.h tail, so the gate ASSERTS it is inert
    # (global grad-norm <= grad_clip) at the gated step — a real parity, not a hollow
    # pass. mamba×neuralgrok is NOT here (no mamba TC neuralgrok tail).
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
    # kernel's [m|v|s_track] against the REAL eager Prodigy's exp_avg/exp_avg_sq/s
    # (run_cell_gate has a prodigy branch: the third buffer is `s`, not `ema`). At
    # step 1 param_init==params ⇒ r=0 ⇒ d stays at d0=1e-6 (matching eager _d_lr=d0),
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
    # step 1 (a SAM step), so the 2nd backward fires. run_cell_gate has a looksam branch:
    #   (1a) params — copy the kernel's OWN sam_dir into opt_ref's sam_direction, inject
    #        the kernel grad g, then opt_ref.step() blends the SAME direction and runs
    #        AdamW: tight (1e-4) APPLY-TAIL parity (the 14/0 apply math on the kernel's
    #        reduced grad + sam_dir). param_tol stays tight — the apply has no NS-style
    #        fp32 accumulation, so a dropped blend / wrong AdamW shows immediately.
    #   (1b) STATE — m/v from the blended grad (tight 1e-4); the `extra` slice (== the
    #        kernel's sam_dir) is checked against an INDEPENDENT bf16-floor reference: the
    #        eager LookSAM.sam_step's 2nd backward (fp32 autograd through the reference
    #        model at the SAME perturbed weights) → sam_dir_eager = g_sam_eager − g. The
    #        kernel's bf16-TC 2nd backward vs the fp32 eager 2nd backward differ only by
    #        the bf16 floor (the SAME DESIGN ≤2e-2 the first-grad adamw gate calibrates),
    #        so sam_dir is checked at 2e-2 — a REAL parity against a real 2nd backward, not
    #        a self-referential pass. A kernel that SKIPPED the SAM phase (sam_dir≈0) FAILS
    #        this (g_sam_eager−g is O(rho)≫0). The 2nd backward's exact correctness is
    #        inherited from the shared adamw grad gate (same fwd/bwd/assembly code), and
    #        (2) A/A/A proves the 2nd pass introduced no race.
    "looksam/decoder": dict(model="decoder", opt="looksam", factory=_looksam_factory,
                            sam_dir_tol=2e-2),
    # looksam (vit — SAM-tier lane): CONVERTED. The IDENTICAL in-kernel P2.4 SAM 2nd
    # backward ported onto the ViT TC kernel (fused_vit_megakernel.cuh): on the step-1
    # SAM step it perturbs p'=p+(rho/‖g‖)·g, runs a FULL SECOND ViT fwd+bwd at p' →
    # g_sam (via vittc_forward_tile/vittc_backward_tile + vittc_dw_*/clspos/lnvec into a
    # SEPARATE sam_grad buffer), writes sam_dir=g_sam−g (persisted in `extra`), restores
    # p. (1a/1b) parity + the sam_dir oracle use the ViT bf16-faithful oracle (gate (A),
    # vit branch above); (2) A/A/A proves the 2nd pass introduced no race (vit's forward
    # is deterministic — vit Prodigy/Muon are A/A/A-green, same fixed reductions).
    "looksam/vit": dict(model="vit", opt="looksam", factory=_looksam_factory,
                        sam_dir_tol=2e-2),
    # supergrok11 (decoder/vit — SAM-tier lane): CONVERTED. The MODEL-COUPLED SAM 2nd backward
    # (P2.4, sharpness=(g_sam−g)²) + the per-TENSOR meta-net mu/gate precompute (P2.45,
    # mu=rescale·phi(g,sharpness), gate=clamp(cos(g,mu),0,1)) are IN-KERNEL phases; the apply
    # tail (sg11_sweep_b_step: smart_grad=g+(1−gate)·alpha·mu, AdamW) runs in P3. The gate
    # (factory: warmup_steps=0 ⇒ ramp=1, rescale=0.1 ⇒ mu!=0) validates FOUR surfaces vs the
    # canonical fp64 math (run_cell_gate's opt=="supergrok11"/"supergrok15" branch):
    #   (A) sharpness — vs the bf16-faithful 2nd backward (g_sam−g)² (the SAME oracle the
    #       looksam sam_dir check uses, squared); a SKIPPED SAM phase (sharpness≈0) fails it.
    #   (B) mu — vs rescale·phi(g, sharpness) (canonical sg11_phi_forward / ref_sg_phi_forward),
    #       on the SAME captured grad + the kernel's OWN sharpness (so the meta-net forward is
    #       validated independent of the bf16 2nd-backward floor).
    #   (1a) params + (1b) m/v — vs ref_sg11_step (smart_grad=g+(1−gate)·alpha·mu, AdamW) with
    #       the kernel's grad/mu/gate, tight 1e-4 (the apply tail math).
    # mamba×supergrok11 is BLOCKED (shared mamba-forward A/A/A race; code-absent in the mamba
    # kernel/launcher). sharpness_tol=2e-2 (the bf16-TC vs bf16-faithful 2nd-backward floor,
    # the SAME design floor as looksam's sam_dir); mu_tol=3e-3 (fp32 phi reorder, SG-family).
    "supergrok11/decoder": dict(model="decoder", opt="supergrok11",
                                factory=_supergrok11_factory,
                                sharpness_tol=2e-2, mu_tol=3e-3),
    "supergrok11/vit": dict(model="vit", opt="supergrok11",
                            factory=_supergrok11_factory,
                            sharpness_tol=2e-2, mu_tol=3e-3),
    # supergrok15 (decoder/vit — SAM-tier lane): CONVERTED. SAME SAM 2nd backward + mu precompute
    # as SG11, but SIMPLER tail: the gate is a HOST SCALAR (sigmoid(accuracy)), NO per-tensor
    # cosine — so the precompute is just mu=rescale·phi(g,sharpness); the apply (sg15_sweep_b_step)
    # does the per-coord alpha clip + smart_grad=g+gate·a·mu + AdamW. Validated vs ref_sg15_step.
    "supergrok15/decoder": dict(model="decoder", opt="supergrok15",
                                factory=_supergrok15_factory,
                                sharpness_tol=2e-2, mu_tol=3e-3),
    "supergrok15/vit": dict(model="vit", opt="supergrok15",
                            factory=_supergrok15_factory,
                            sharpness_tol=2e-2, mu_tol=3e-3),
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
                          sam_dir_tol=2e-2),
    # supergrok11/15 (mamba — wave-4 mamba SG lane): CONVERTED via the FEATURE PORT (the
    # decoder/vit twins ported onto fused_mamba_megakernel.cuh + mega_mamba_real_adamw_tc_
    # launcher.cu): the P2.4 SAM 2nd backward extends to SG (sharpness=(g_sam−g)²) + the
    # per-tensor meta-net mu precompute (P2.45/P3) + the SG11 cosine gate run as in-kernel
    # phases; the launcher gained the OptId::SuperGrok11/15 cases (opt_id 8/9). The shared-
    # mamba-forward A/A/A race is FIXED (commit 0b57f7e — the LookSAM mamba instantiation
    # proved the SAM double-forward is race-free), so the (2) A/A/A determinism check below
    # VERIFIES SG11/15 are bit-exact too. Same 4-surface validation as decoder/vit (sharpness
    # vs the bf16-faithful 2nd backward, mu vs the canonical phi, 1a/1b apply tail). HONEST
    # CAVEAT: SG11/15 mamba reach L3-TC + single-step parity but WON'T GROK on L3 (the meta-net
    # is untrained — a separate owner-approved host-training task, NOT done here).
    "supergrok11/mamba": dict(model="mamba", opt="supergrok11",
                              factory=_supergrok11_factory,
                              sharpness_tol=2e-2, mu_tol=3e-3),
    "supergrok15/mamba": dict(model="mamba", opt="supergrok15",
                              factory=_supergrok15_factory,
                              sharpness_tol=2e-2, mu_tol=3e-3),
    # supergrok2 (decoder/vit — the LAST/hardest cell): CONVERTED via the DEDICATED
    # ops.sg2_fused_step entry. The FULL CSA/HCA/PEER/GRU meta-net AS the optimizer phase
    # (P3-SG2): in-kernel SEGMENTED SORT (STAGE -1, index-tie-break strategy A) → S0..S5,
    # reading st.sharpness from the SAM 2nd backward (P2.4, shared with SG11/15). The SG2
    # gate (tests/hw/_sg2_l3tc_gate.run_sg2_gate) validates FOUR surfaces: (B1) single-step
    # vs ops.supergrok2_batched_step (the per-op ORACLE, SAME low-rank indexer) max|Δ{param,
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
    # __SG2_MAMBA_TEST_CELL__
}

# BLOCKED cells — kept here (commented) with the state-gate evidence that blocks them,
# so the reason is reproducible. NOT registered in _FUSED_L3_REAL; the gate's
# has_l3_real precondition would (correctly) refuse to run them as "converted".
#  * grokadamw/vit is now CONVERTED (wave-2 vit lane — registered in _CELLS above): the
#    identical P2.5 global-norm clip + P3 per-tensor β1 are wired into the vit TC kernel
#    (fused_vit_megakernel_tc), the vit launcher already routes opt_id=3, and γ/grad_clip
#    thread via FusedScalars. So _BLOCKED_EVIDENCE is now empty for this lane.
_BLOCKED_EVIDENCE = {}


def _build_cell(model, seed=42):
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


def _run_sg_cell_gate(cell_key, spec, g, c, m, data, dev, canon, opt_obj, grad,
                      loss, p_after, total, named0, cache, verbose):
    """Dedicated SuperGrok11/15 gate: validate the kernel vs the CANONICAL fp64 math its
    apply_optimizer<SG> calls (ref_sg11_step/ref_sg15_step + ref_sg_phi_forward), plus the
    SAM 2nd backward (sharpness) vs the bf16-faithful oracle. Returns (ok, detail)."""
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
    sam_on = float(scalars.get("looksam_sam", 0.0))
    beta1, beta2 = float(scalars["beta1"]), float(scalars["beta2"])
    lr, eps, wd = float(scalars["lr"]), float(scalars["eps"]), float(scalars["weight_decay"])
    alpha_max = float(scalars.get("alpha_max", alpha))

    # Per-tensor (name, offset, numel) in the flat layout.
    off = 0; layout = []
    for n_, p in named0:
        layout.append((n_, off, p.numel(), p.shape)); off += p.numel()

    # ── (A) sharpness vs the bf16-faithful 2nd backward (g_sam−g)². Liveness: a real SAM
    # step makes sharpness O((ron·‖∇‖)²) ≫ 0. The factory uses k=… so step 1 IS a SAM step.
    g_r, c_r, m_ref, data_r, dev_r = _build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    sharp_ref = _sg_bf16_sharpness_oracle(model, named_ref, tx, ty, grad, rho, total, dev)
    assert sam_on != 0.0, f"{cell_key}: looksam_sam==0 — the SAM 2nd backward did not run at step 1"
    assert sharp_ref.abs().max().item() > 1e-8, (
        f"{cell_key}: oracle sharpness is ~0 (the 2nd backward produced no signal) — vacuous gate")
    sh_abs = (k_sharp - sharp_ref).abs().max().item()
    sh_den = sharp_ref.abs().max().item() + 1e-30
    sharp_rel = sh_abs / sh_den
    sharp_tol = spec.get("sharpness_tol", 2e-2)
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
    # (1a/1b) isolate; (B) already validated mu separately). m/v start at 0 (zero-init cache).
    p_ref = torch.empty(total, device=dev, dtype=torch.float64)
    m_ref_f = torch.empty(total, device=dev, dtype=torch.float64)
    v_ref_f = torch.empty(total, device=dev, dtype=torch.float64)
    p_before_t = torch.cat([p.data.reshape(-1).double() for _, p in named_ref])
    for (n_, o, k, sh) in layout:
        gt = grad[o:o + k].double()
        mut = k_mu[o:o + k].double()
        pt = p_before_t[o:o + k]
        zt = torch.zeros(k, dtype=torch.float64, device=dev)
        if opt == "supergrok11":
            # per-tensor cosine gate (the kernel's P2.45 gate): clamp(<g,mu>/sqrt(|g|²|mu|²+1e-12),0,1).
            num = (gt * mut).sum()
            den = torch.sqrt((gt * gt).sum() * (mut * mut).sum() + 1e-12)
            gate_t = float(torch.clamp(num / den, 0.0, 1.0)) if den > 0 else 0.0
            pn, mn, vn = ref_sg11_step(pt, gt, zt, zt, mut, gate=gate_t, alpha=alpha,
                                       lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd, t=1)
        else:
            pn, mn, vn = ref_sg15_step(pt, gt, zt, zt, mut, gate_global=gate_global,
                                       alpha_base=alpha, alpha_max=alpha_max, lr=lr,
                                       beta1=beta1, beta2=beta2, eps=eps, wd=wd, t=1)
        p_ref[o:o + k] = pn; m_ref_f[o:o + k] = mn; v_ref_f[o:o + k] = vn

    def _rel(a, b):
        d = (a.double() - b).abs().max().item(); return d / (b.abs().max().item() + 1e-30)
    param_rel = _rel(p_after, p_ref)
    m_rel = _rel(k_m, m_ref_f)
    v_rel = _rel(k_v, v_ref_f)
    param_tol = spec.get("param_tol", 1e-4)
    param_ok = param_rel < param_tol
    state_ok = (m_rel < 1e-4 and v_rel < 1e-4)
    if verbose:
        print(f"  (1a) params vs canonical SG: max-rel={param_rel:.3e} (tol {param_tol:.0e}) "
              f"{'OK' if param_ok else 'FAIL'}  loss={loss:.5f}", flush=True)
        print(f"  (1b) STATE vs canonical SG: m={m_rel:.3e} v={v_rel:.3e} (tol 1e-4) "
              f"{'OK' if state_ok else 'FAIL'}", flush=True)
        print(f"  (A) sharpness vs bf16-faithful 2nd backward: rel={sharp_rel:.3e} "
              f"(tol {sharp_tol:.0e}, bf16-TC vs bf16-faithful floor) "
              f"{'OK' if sharp_ok else 'FAIL'}", flush=True)
        print(f"  (B) mu vs rescale·phi(g,sharpness): rel={mu_rel:.3e} (tol {mu_tol:.0e}) "
              f"{'OK' if mu_ok else 'FAIL'}  [rescale={float(rescale):.3g} alpha={alpha:.3g} "
              f"gate_global={gate_global:.3g}]", flush=True)
    tail_ok = param_ok and state_ok and sharp_ok and mu_ok

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


def run_cell_gate(cell_key, verbose=True):
    """Run gates (1)+(2) for one converted cell. Returns (ok, detail)."""
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)
    spec = _CELLS[cell_key]
    model, opt = spec["model"], spec["opt"]
    # ── SuperGrok2 dedicated gate (the LAST/hardest cell). SG2's optimizer phase is the
    # FULL CSA/HCA/PEER/GRU meta-net (in-kernel segmented sort + SAM 2nd backward +
    # sg2_meta_stages), validated against the per-op ORACLE (ops.supergrok2_batched_step,
    # the SAME low-rank indexer) — NOT the eager opt.step() (its bilevel/sam/ramp binding
    # is not the L3 apply). tests/hw/_sg2_l3tc_gate owns the (B1)+(A/A/A)+(tie-probe)+(N>64
    # CSA fidelity probe) verdict; short-circuit run_cell_gate to it.
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
    # The SG path does NOT run the eager opt.step() comparison (its bilevel/ramp binding is
    # not the L3 apply): it validates the kernel against the CANONICAL fp64 math the kernel's
    # apply_optimizer<SG> calls — the same single-source as ref_sg11_step/ref_sg15_step — plus
    # the SAM 2nd backward (sharpness) vs the bf16-faithful oracle and mu vs rescale·phi. Short-
    # circuits run_cell_gate (it owns its own (1a/1b)+(A/B)+(2) verdict).
    if opt in ("supergrok11", "supergrok15"):
        ok, detail = _run_sg_cell_gate(cell_key, spec, g, c, m, data, dev, canon,
                                       opt_obj, grad, loss, p_after, total,
                                       named0, cache, verbose)
        return ok, detail

    # REFERENCE = the REAL eager optimizer (not a header re-transcription). Build a
    # FRESH model with the SAME init (same seed), INJECT the kernel's TC-reduced grad
    # into each p.grad (so both sides consume the BYTE-IDENTICAL gradient — the bf16-TC
    # grad — isolating the optimizer TAIL math), then run ONE real opt.step() from
    # zero state. A mechanism the real optimizer has but apply_optimizer<Opt> drops
    # (grad_clip, per-layer schedule, adaptive alpha) SHOWS UP here as a param mismatch.
    g_r, c_r, m_ref, data_r, dev_r = _build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    # confirm the fresh model's init matches p_before (same seed) — the reference is
    # only valid if it starts from the kernel's input params.
    p_ref_before = torch.cat([p.data.reshape(-1) for _, p in named_ref])
    assert torch.equal(p_ref_before, p_before), \
        f"{cell_key}: reference model init != kernel init (seed mismatch)"
    opt_ref = spec["factory"](g_r, m_ref.parameters(), c_r)
    # NEURALGROK: the kernel read the psi-net weights the host scattered from
    # opt_obj.psi_pack() into the `extra` slice (fused_train_step). The eager
    # reference must therefore hold the SAME amplifier weights, or it would apply a
    # DIFFERENT psi(|g|) and the state would diverge for a reason that is NOT a
    # dropped mechanism. opt_obj and opt_ref draw independent random amplifier inits,
    # so copy opt_obj's amplifier into opt_ref and mark its kernel-weight snapshot
    # dirty (so the eager step re-extracts the copied weights). This isolates the
    # check to the TAIL+psi MATH (the gate's purpose), exactly as the SAME-grad
    # injection isolates it from the bwd.
    if opt == "neuralgrok":
        opt_ref.amplifier.load_state_dict(opt_obj.amplifier.state_dict())
        opt_ref.mark_amplifier_dirty()
    # Scatter the kernel's flat TC grad into the reference params' .grad (layout order).
    off = 0
    for _, p in named_ref:
        n = p.numel()
        p.grad = grad[off:off + n].reshape(p.shape).clone()
        off += n
    # NEURALGROK clip-inertness GUARD (no-suppression honesty): the eager NeuralGrok
    # binding applies a GLOBAL grad-norm clip (helpers.h clip_grad_norms_device_side)
    # to grad_clip=1.0 BEFORE the apply; the kernel's neuralgrok.h tail does NOT clip.
    # The two paths match ONLY when the clip is inert (global grad-norm <= grad_clip).
    # Assert it here so a step where the clip WOULD fire fails LOUD instead of hollow-
    # passing on a coincidentally-small step-1 norm. (grad_clip<=0 disables the clip,
    # so the guard is vacuously satisfied there — the kernel is then exactly faithful.)
    if opt == "neuralgrok":
        gclip = float(opt_ref.param_groups[0].get("grad_clip", 0.0))
        if gclip > 0.0:
            gnorm = grad.detach().double().norm().item()  # global L2 over all tensors
            assert gnorm <= gclip + 1e-6, (
                f"{cell_key}: eager NeuralGrok grad-norm clip would FIRE "
                f"(global grad-norm {gnorm:.4f} > grad_clip {gclip}); the kernel's "
                f"neuralgrok.h tail does not clip, so this step's parity would be a "
                f"hollow pass. The clip is the one eager-binding mechanism the "
                f"single-launch tail cannot carry — at the gated step it must be inert.")
        # The decoder/vit TC launcher uses raw cudaMalloc scratch (DecTcLauncherScratch),
        # so the caching-allocator layout differs after the kernel ran. Fully sync the
        # device before the eager reference's per-op neuralgrok kernel so its psi-weight
        # staging is not racing the just-finished L3 launch's teardown.
        torch.cuda.synchronize()
    # LOOKSAM: the kernel ran the in-kernel SAM 2nd backward (P2.4) on this (step-1) SAM
    # step, writing sam_dir = g_sam − g into the `extra` state slice and BLENDING
    # g_adj=(1−α)g+α·sam_dir in the apply. The gate validates BOTH surfaces:
    #   (A) sam_dir parity — vs the bf16-FAITHFUL fp64 oracle's OWN 2nd backward (the
    #       SAME instrument test_tc_single_step_grad_parity uses for the first grad, NOT
    #       a fp32-autograd reference: fp32-autograd carries a ~6.6e-2 bf16-storage gap
    #       that is WRONG for a bf16-TC kernel — measured: kernel-vs-bf16faithful 1.8e-2,
    #       eager-fp32-vs-bf16faithful 6.6e-2, so the kernel is CLOSER to the bf16 truth
    #       than fp32 is). We perturb the oracle params by ron·g (ron=rho/‖g‖_global, the
    #       SAME global-norm perturb the kernel + binding's looksam_perturb_all use, with
    #       g the kernel's reduced grad) and form sam_dir_bf = g_sam_bf − g_bf. Compared
    #       at the DESIGN bf16 floor (sam_dir_tol=2e-2). A SKIPPED SAM phase (sam_dir≈0)
    #       fails it (‖sam_dir_bf‖ is O(ron·‖∇‖)≫0). DECODER-specific (the oracle is the
    #       decoder one); vit/mamba use their own bf16-faithful oracle when ported.
    #   (B) apply-tail parity — set opt_ref's sam_direction to the KERNEL's sam_dir so
    #       opt_ref.step() blends the SAME direction the kernel used; this isolates (1a)
    #       params + (1b) m/v to the apply math at tight tol (1e-4), free of the 2nd-
    #       backward bf16 gap. The injected p.grad already holds the kernel's g.
    looksam_sam_dir_rel = None
    if opt == "looksam":
        kstate0 = cache[canon]["state"]
        k_sam = kstate0[2 * total:3 * total]           # the kernel's cached sam_dir
        # (A) bf16-faithful sam_dir reference (per-model oracle, perturbed by ron·g).
        #     The SAME bf16-faithful fp64 oracle each model's first-grad gate uses
        #     (test_<model>_tc), run TWICE (at p and at p'=p+ron·g) → sam_dir_bf =
        #     g_sam_bf − g_bf, compared to the kernel's sam_dir at the bf16 floor.
        if model == "decoder":
            from tests.hw.test_decoder_tc import _bf16_faithful_oracle
            B = int(tx.shape[0]); B16 = B - (B % 16)
            tok_o = tx[:B16].to(torch.long); tgt_o = ty[:B16].to(torch.long)
            named_d = {n: p.detach() for n, p in named_ref}
            _lo, grads_o = _bf16_faithful_oracle(named_d, tok_o, tgt_o)
            gn = torch.sqrt(sum((gv.double() ** 2).sum() for gv in grads_o.values())).item()
            rho = float(opt_ref.param_groups[0]["rho"])
            ron = rho / gn
            # perturb each param by ron·g (g = the kernel's reduced grad, layout order).
            off = 0; named_pert = {}
            for n_, p in named_ref:
                k = p.numel()
                named_pert[n_] = (p.detach().double()
                                  + ron * grad[off:off + k].reshape(p.shape).double())
                off += k
            _lo2, grads_o2 = _bf16_faithful_oracle(named_pert, tok_o, tgt_o)
            r_sam = torch.cat([(grads_o2[n_] - grads_o[n_]).reshape(-1).double()
                               for n_, _ in named_ref]).float().to(dev)
        elif model == "vit":
            # ViT bf16-faithful oracle (test_vit_tc._bf16_faithful_oracle): input is
            # FLOAT image patches [B,16,49] (tx == data[0]), targets [B]. B16-truncate
            # to match the kernel's B%16 batch (fused_train_step truncates the same way).
            from tests.hw.test_vit_tc import _bf16_faithful_oracle as _vit_bf16_oracle
            B = int(tx.shape[0]); B16 = B - (B % 16)
            patches_o = tx[:B16].detach().cpu().double()
            tgt_o = ty[:B16].detach().cpu().to(torch.long)
            named_d = {n: p.detach().cpu() for n, p in named_ref}
            _lo, grads_o = _vit_bf16_oracle(named_d, patches_o, tgt_o)
            gn = torch.sqrt(sum((gv.double() ** 2).sum() for gv in grads_o.values())).item()
            rho = float(opt_ref.param_groups[0]["rho"])
            ron = rho / gn
            # perturb each param by ron·g (g = the kernel's reduced grad, layout order).
            off = 0; named_pert = {}
            for n_, p in named_ref:
                k = p.numel()
                named_pert[n_] = (p.detach().cpu().double()
                                  + ron * grad[off:off + k].reshape(p.shape).cpu().double())
                off += k
            _lo2, grads_o2 = _vit_bf16_oracle(named_pert, patches_o, tgt_o)
            r_sam = torch.cat([(grads_o2[n_] - grads_o[n_]).reshape(-1).double()
                               for n_, _ in named_ref]).float().to(dev)
        elif model == "mamba":
            # Mamba bf16-faithful oracle (test_mamba_tc._bf16_faithful_mamba_oracle):
            # input is int tokens [B,kSeq] (tx == data[0]) like the decoder, targets [B].
            # B16-truncate to match the kernel's B%16 batch (fused_train_step truncates
            # the same way; the mamba TC launcher requires B%16==0).
            from tests.hw.test_mamba_tc import _bf16_faithful_mamba_oracle as _mb_bf16_oracle
            B = int(tx.shape[0]); B16 = B - (B % 16)
            tok_o = tx[:B16].detach().cpu().to(torch.long)
            tgt_o = ty[:B16].detach().cpu().to(torch.long)
            named_d = {n: p.detach().cpu() for n, p in named_ref}
            _lo, grads_o = _mb_bf16_oracle(named_d, tok_o, tgt_o)
            gn = torch.sqrt(sum((gv.double() ** 2).sum() for gv in grads_o.values())).item()
            rho = float(opt_ref.param_groups[0]["rho"])
            ron = rho / gn
            # perturb each param by ron·g (g = the kernel's reduced grad, layout order).
            off = 0; named_pert = {}
            for n_, p in named_ref:
                k = p.numel()
                named_pert[n_] = (p.detach().cpu().double()
                                  + ron * grad[off:off + k].reshape(p.shape).cpu().double())
                off += k
            _lo2, grads_o2 = _mb_bf16_oracle(named_pert, tok_o, tgt_o)
            r_sam = torch.cat([(grads_o2[n_] - grads_o[n_]).reshape(-1).double()
                               for n_, _ in named_ref]).float().to(dev)
        else:
            raise NotImplementedError(
                f"{cell_key}: looksam sam_dir reference oracle not wired for {model} "
                f"(only decoder/vit/mamba bf16-faithful oracles are available here).")
        sd_abs = (k_sam - r_sam).abs().max().item()
        sd_den = r_sam.abs().max().item() + 1e-30
        looksam_sam_dir_rel = sd_abs / sd_den
        # Liveness: a real SAM step makes sam_dir O(ron·‖∇‖) ≫ 0 — a skipped phase is ~0.
        assert r_sam.abs().max().item() > 1e-6, (
            f"{cell_key}: oracle sam_dir is ~0 (the 2nd backward produced no direction) "
            f"— the gate's sam_dir parity would be vacuous.")
        # (B) set opt_ref's sam_direction to the KERNEL's sam_dir for the apply parity.
        # LookSAM.step()'s _group_cache only INITIALISES state[p] when len(state)==0, so
        # we must seed the FULL state (exp_avg/exp_avg_sq/step/sam_direction) here — else
        # adding only sam_direction makes len(state)>0 and step() KeyErrors on exp_avg.
        # exp_avg/exp_avg_sq start at ZERO (the kernel's zero-init m/v cache), step=0.
        off = 0
        for _, p in named_ref:
            n = p.numel()
            stt = opt_ref.state[p]
            stt["step"] = 0
            stt["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            stt["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            stt["sam_direction"] = k_sam[off:off + n].reshape(p.shape).clone().float()
            off += n
        # Invalidate the static cache so step() rebuilds it from the seeded state (the
        # factory may have warmed it; we want our seeded exp_avg/sam_direction used).
        opt_ref._static_cache = {}
    opt_ref.step()
    p_ref = torch.cat([p.data.reshape(-1) for _, p in named_ref])

    # The kernel and the real optimizer consumed the SAME grad; the only legitimate
    # difference is fp32 rounding order in the elementwise tail. A DROPPED mechanism
    # (e.g. grad_clip scaling every grad, or a per-layer beta1) makes this large.
    abs_err = (p_after - p_ref).abs().max().item()
    denom = p_ref.abs().max().item() + 1e-30
    rel_err = abs_err / denom
    # Per-cell PARAM tolerance. Default 1e-4 (fp32 reorder; a dropped mechanism blows
    # past it). MUON's 2D-weight params go through a 5-iteration Newton-Schulz whose
    # fp32 vs the eager fp32 NS accumulate to ~2e-3 rel (INTEGRATION-OPTSTAGES §8 sets
    # the muon NS oracle tol at <2e-3, NOT 1e-9) — so muon uses 2e-3 for (1a). The (1b)
    # STATE check stays TIGHT (1e-4): the momentum buffer (m) is buf=μ·buf+g, NO NS, so
    # it must match eager exactly — that is the load-bearing "runs the real optimizer"
    # check, and a dropped/garbage momentum still fails it at 1e-4.
    param_tol = spec.get("param_tol", 1e-4)
    param_ok = rel_err < param_tol
    if verbose:
        print(f"  (1a) params vs REAL eager: max|Δp|={abs_err:.3e} rel={rel_err:.3e} "
              f"(tol {param_tol:.0e}) {'OK' if param_ok else 'FAIL'}  loss={loss:.5f}", flush=True)

    # ── (1b) STATE vs REAL eager — THE DECISIVE CHECK. At step 1 every Adam-family
    # tail collapses to ≈sign(g_amp) in the PARAM update (m/bc1=g_amp, sqrt(v/bc2)=
    # |g_amp|), so (1a) is BLIND to per-layer beta1 (gamma), amplification magnitude,
    # and STATE INITIALIZATION — all of which live in the optimizer state, not the
    # sign. So we compare the kernel's post-step state slices [m|v|ema] (in the cache
    # buffer) against the REAL optimizer's opt.state[p]. A state mismatch == the kernel
    # is NOT running the real optimizer (e.g. grokfast's ema cold-start: real inits
    # ema=g0, the kernel inits ema=0 → the ema slice diverges 50x even when params
    # match). This is what makes the single-step verdict mean "runs the real optimizer".
    kstate = cache[canon]["state"]              # flat [m|v|extra] (3*total)+loss
    k_m = kstate[0:total]; k_v = kstate[total:2 * total]; k_ema = kstate[2 * total:3 * total]
    # Gather the real optimizer's per-parameter state into flat buffers (layout order).
    r_m = torch.zeros(total, device=dev); r_v = torch.zeros(total, device=dev)
    r_ema = torch.zeros(total, device=dev)
    have_v = have_ema = False
    off = 0
    for _, p in named_ref:
        n = p.numel(); stt = opt_ref.state.get(p, {})
        # MUON: the eager optimizer keeps the 2D-matrix momentum under the
        # "momentum_buffer" key (muon.py:196, the Newton-Schulz group), NOT "exp_avg"
        # — only the 1D AdamW group uses "exp_avg". The kernel stores BOTH in the same
        # m-slice (st.exp_avg): the running momentum buf=μ·buf+g for the 2D weights and
        # the AdamW m for the 1D weights. So the (1b) m-reference must read whichever
        # key the eager optimizer actually populated for THIS tensor. Without this, the
        # 11 NS matrices read 0 (no "exp_avg" key) and the check reports a phantom 21.x
        # mismatch even though the kernel's 2D momentum is bit-identical to eager. This
        # makes (1b) RIGOROUS for Muon (it now validates the real NS-group momentum),
        # not weaker — a wrong-magnitude buf (which NS would orthogonalize to the same
        # direction, hiding in (1a) params) fails HERE. Non-Muon cells lack the
        # "momentum_buffer" key, so the exp_avg branch is unchanged for them.
        if "momentum_buffer" in stt:
            r_m[off:off + n].copy_(stt["momentum_buffer"].reshape(-1))
        elif "exp_avg" in stt:
            r_m[off:off + n].copy_(stt["exp_avg"].reshape(-1))
        if "exp_avg_sq" in stt:
            r_v[off:off + n].copy_(stt["exp_avg_sq"].reshape(-1)); have_v = True
        if "ema" in stt:
            r_ema[off:off + n].copy_(stt["ema"].reshape(-1)); have_ema = True
        # PRODIGY: the kernel's third state slice (extra) holds Prodigy's `s`
        # trajectory accumulator (s_track), not an EMA. The REAL eager Prodigy keeps
        # it under the "s" key. Map it into the r_ema comparison slot so the (1b)
        # STATE check validates the kernel's s_track += d·g against eager (the
        # apply-tail's third write — load-bearing for the d-trajectory at step≥2).
        if "s" in stt:
            r_ema[off:off + n].copy_(stt["s"].reshape(-1)); have_ema = True
        off += n
    def _rel(a, b):
        d = (a - b).abs().max().item(); return d / (b.abs().max().item() + 1e-30)
    # NEURALGROK (1b) reference = the CANONICAL neuralgrok.h math (the gate docstring's
    # defined reference), NOT the live eager .state. The eager per-op neuralgrok kernel
    # has a cross-cell contamination bug (see _neuralgrok_canonical_mv): in-process,
    # after another decoder/vit cell, its g_amp magnitude drifts (eager-m vs canonical
    # ~1.7e-2) though it is BIT-EXACT to the kernel in a fresh process. The L3-TC kernel
    # under gate matches the canonical math to ~2.7e-7 regardless. We swap r_m/r_v to the
    # canonical reference and ALSO report the eager-vs-canonical delta so the eager
    # contamination is visible, not hidden. (1a) still checks live-eager params.
    if opt == "neuralgrok":
        eager_m_rel = _rel(k_m, r_m)
        eager_v_rel = _rel(k_v, r_v) if have_v else 0.0
        r_m, r_v = _neuralgrok_canonical_mv(grad, opt_obj, step=1)
        have_v = True
    m_rel = _rel(k_m, r_m)
    v_rel = _rel(k_v, r_v) if have_v else 0.0
    ema_rel = _rel(k_ema, r_ema) if have_ema else None
    # The kernel and real opt consumed the SAME grad; their state must match to fp32
    # reorder tol. ema only checked when the optimizer HAS an ema buffer.
    state_ok = (m_rel < 1e-4 and v_rel < 1e-4
                and (ema_rel is None or ema_rel < 1e-4))
    # LOOKSAM: the `extra` slice is the SAM direction (sam_dir=g_sam−g), NOT an EMA/s —
    # so it is NOT in the m/v/ema state check above. It is validated SEPARATELY against
    # the INDEPENDENT eager 2nd backward (looksam_sam_dir_rel, computed before step())
    # at the bf16 floor (sam_dir_tol). This is the load-bearing check for the new SAM
    # machinery — fold it into state_ok so a wrong/skipped sam_dir FAILS the gate.
    sam_ok = True
    if opt == "looksam":
        sam_tol = spec.get("sam_dir_tol", 2e-2)
        sam_ok = (looksam_sam_dir_rel is not None and looksam_sam_dir_rel < sam_tol)
        state_ok = state_ok and sam_ok
    if verbose:
        ema_s = f"ema={ema_rel:.3e}" if ema_rel is not None else "ema=n/a"
        ref_s = "canonical neuralgrok.h" if opt == "neuralgrok" else "REAL eager"
        print(f"  (1b) STATE vs {ref_s}: m={m_rel:.3e} v={v_rel:.3e} {ema_s} "
              f"(tol 1e-4) {'OK' if state_ok else 'FAIL — kernel state != reference'}",
              flush=True)
        if opt == "looksam":
            print(f"  (1b-sam) sam_dir vs bf16-faithful 2nd backward: rel={looksam_sam_dir_rel:.3e} "
                  f"(tol {spec.get('sam_dir_tol', 2e-2):.0e}, kernel bf16-TC vs bf16-faithful "
                  f"fp64 oracle = the DESIGN bf16 floor) "
                  f"{'OK' if sam_ok else 'FAIL — kernel sam_dir != bf16-faithful 2nd backward'}",
                  flush=True)
        if opt == "neuralgrok":
            print(f"       [diag] eager per-op kernel vs canonical: m={eager_m_rel:.3e} "
                  f"v={eager_v_rel:.3e} (large => eager cross-cell contamination, "
                  f"out-of-scope CUDA-kernel bug; the L3-TC kernel above is clean)",
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


@pytest.mark.hw
@pytest.mark.skipif(not _gpu_ready(), reason="needs built extension on GPU")
@pytest.mark.parametrize("cell", list(_CELLS))
def test_l3tc_cell_gate(cell):
    ok, detail = run_cell_gate(cell, verbose=True)
    assert ok, f"{cell} L3-TC gate failed: {detail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="", help="single cell e.g. lion/decoder")
    args = ap.parse_args()
    if not _gpu_ready():
        print("SKIP: no built extension / GPU", flush=True)
        return
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
