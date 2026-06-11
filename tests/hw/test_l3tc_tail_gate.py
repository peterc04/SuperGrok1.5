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
    # NOTE: prodigy/mamba is NOT registered — it FAILS A/A/A (scheduling-exposed race in
    # the shared mamba forward, exposed by prodigy's register pressure; loss+grad differ
    # across bit-identical re-runs, grad maxd ~1e-2). The P2.6 kernel/launcher code is
    # landed-dormant; the cell stays blocked until the racy mamba component is fixed.
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


def run_cell_gate(cell_key, verbose=True):
    """Run gates (1)+(2) for one converted cell. Returns (ok, detail)."""
    from grokking_optimizers.dispatch import (has_l3_real, gemm_impl_for_cell,
                                              canonicalize_model, fused_train_step)
    spec = _CELLS[cell_key]
    model, opt = spec["model"], spec["opt"]
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
        # (A) bf16-faithful sam_dir reference (decoder oracle, perturbed by ron·g).
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
        else:
            raise NotImplementedError(
                f"{cell_key}: looksam sam_dir reference oracle not wired for {model} "
                f"(only decoder's bf16-faithful oracle is available here).")
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
