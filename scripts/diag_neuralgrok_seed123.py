# Diagnostic: classify the neuralgrok/decoder (1a) FAIL at GATE_SEED=123 as
#   (a) benign-within-bf16  vs  (b) real kernel-tail gap.
# READ-ONLY harness reuse: builds the SAME cell the gate builds, runs ONE real
# L3-TC step (kernel-under-test, megakernel apply_optimizer<NeuralGrok>, H=16),
# the SAME eager reference the gate uses (opt_ref.step() -> per-op
# neuralgrok_fused_step), AND a FULL-fp64 recomputation of the neuralgrok update
# on the kernel's own reduced grad with the TRUE H=16 amplifier. Prints, for the
# single worst-Δ coordinate: param name, abs Δ(kernel vs eager), reference
# magnitude there, and BOTH kernel & eager vs the fp64 oracle. The fp64 oracle is
# the arbiter: if the KERNEL matches it to bf16 roundoff while the EAGER does not,
# the kernel is correct and the FAIL is the gate's reference (classification (a),
# the 1e-4 pure-rel tol with no abs floor). If the KERNEL diverges from the fp64
# oracle, that is a real tail bug (classification (b)).
#
#   Run:  cd /workspace/SuperGrok1.5 && GATE_SEED=123 PYTHONPATH=. python - < scripts/diag_neuralgrok_seed123.py
#   (or:  GATE_SEED=123 PYTHONPATH=. python scripts/diag_neuralgrok_seed123.py)
import os
os.environ.setdefault("GATE_SEED", "123")   # the failing seed (also try 7; 42 passes)

import torch
from tests.hw.test_l3tc_tail_gate import _build_cell, _neuralgrok_factory
from grokking_optimizers.dispatch import (canonicalize_model, fused_train_step,
                                          has_l3_real, gemm_impl_for_cell)

MODEL = "decoder"
SEED = int(os.environ["GATE_SEED"])
print(f"=== neuralgrok/{MODEL} (1a) diagnostic @ GATE_SEED={SEED} ===")


def _flat_named(m):
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    return named, sum(p.numel() for _, p in named)


def _psi_fp64_H16(abs_g, W1, b1, W2, b2):
    """The EXACT psi the deployed kernel (kPsiHidden=16) computes, in fp64:
       psi = sum_{j<16} W2[j]*relu(W1[j]*|g| + b1[j]) + b2.
    W1=[16,1], b1=[16], W2=[1,16], b2=[1] (the amplifier's get_weights layout)."""
    H = 16
    w1 = W1.reshape(-1).double()          # [16]
    bb1 = b1.reshape(-1).double()         # [16]
    w2 = W2.reshape(-1).double()          # [16]
    bb2 = float(b2.reshape(-1)[0])
    hid = torch.relu(abs_g.unsqueeze(1) * w1.reshape(1, H) + bb1.reshape(1, H))  # [N,16]
    return (hid * w2.reshape(1, H)).sum(1) + bb2                                  # [N]


def _neuralgrok_fp64_params(p_before, grad, opt_obj):
    """FULL-fp64 neuralgrok step-1 update on the kernel's reduced grad, TRUE H=16
    psi (the algorithm's documented contract — neuralgrok.h, kPsiHidden=16). Zero-
    init moments. Returns p_after_fp64 (flat fp64). This is the ARBITER both kernel
    and eager are scored against."""
    grp = opt_obj.param_groups[0]
    alpha = float(grp["alpha"]); beta = float(grp["beta"])
    b1, b2 = float(grp["betas"][0]), float(grp["betas"][1])
    lr = float(grp["lr"]); eps = float(grp["eps"]); wd = float(grp["weight_decay"])
    bc1 = 1.0 - b1 ** 1; bc2 = 1.0 - b2 ** 1
    W1, b1w, W2, b2w = opt_obj.amplifier.get_weights()
    g = grad.double()
    psi = _psi_fp64_H16(g.abs(), W1.to(g.device), b1w.to(g.device),
                        W2.to(g.device), b2w.to(g.device))
    g_amp = (psi * alpha + beta) * g
    m = (1.0 - b1) * g_amp
    v = (1.0 - b2) * g_amp * g_amp
    update = (m / max(bc1, 1e-30)) / (torch.sqrt(v / max(bc2, 1e-30)) + eps)
    p = p_before.double()
    return p - lr * (update + wd * p), g_amp


# ── 1) KERNEL-UNDER-TEST: one production L3-TC step (megakernel, H=16). ──────
g, c, m, data, dev = _build_cell(MODEL, seed=SEED)
canon = canonicalize_model(MODEL)
assert has_l3_real(canon, "neuralgrok"), "neuralgrok/decoder not L3-REAL"
assert gemm_impl_for_cell(canon, "neuralgrok", "bf16") == "wgmma"
tx, ty = data[0], data[1]
opt_obj = _neuralgrok_factory(g, m.parameters(), c)
named0, total = _flat_named(m)
p_before = torch.cat([p.data.reshape(-1) for _, p in named0]).clone()
cache = {}
loss, grad = fused_train_step(canon, "neuralgrok", m, opt_obj, tx, ty,
                              state_cache=cache, step=1, return_grad=True,
                              gemm_impl="wgmma")
p_after_kernel = torch.cat([p.data.reshape(-1) for _, p in named0]).clone()

# global grad-norm clip inertness (the gate asserts this; report it, do not abort)
gclip = float(opt_obj.param_groups[0].get("grad_clip", 0.0))
gnorm = grad.detach().double().norm().item()
print(f"loss={loss:.5f}  global grad-norm={gnorm:.4f}  grad_clip={gclip}  "
      f"clip {'INERT' if (gclip <= 0 or gnorm <= gclip) else 'WOULD FIRE'}")

# ── 2) EAGER REFERENCE: the gate's reference — fresh model, same init, inject the
#       kernel's reduced grad, copy opt_obj's amplifier, one opt_ref.step() (per-op
#       neuralgrok_fused_step kernel, which is compiled at NG_H=64). ─────────────
g_r, c_r, m_ref, data_r, dev_r = _build_cell(MODEL, seed=SEED)
named_ref, _ = _flat_named(m_ref)
p_ref_before = torch.cat([p.data.reshape(-1) for _, p in named_ref])
assert torch.equal(p_ref_before, p_before), "ref init != kernel init (seed mismatch)"
opt_ref = _neuralgrok_factory(g_r, m_ref.parameters(), c_r)
opt_ref.amplifier.load_state_dict(opt_obj.amplifier.state_dict())
opt_ref.mark_amplifier_dirty()
off = 0
for _, p in named_ref:
    n = p.numel(); p.grad = grad[off:off + n].reshape(p.shape).clone(); off += n
torch.cuda.synchronize()
opt_ref.step()
p_after_eager = torch.cat([p.data.reshape(-1) for _, p in named_ref]).clone()

# ── 3) FULL-fp64 ARBITER: neuralgrok step-1 on the kernel's reduced grad, H=16. ─
p_after_fp64, g_amp_fp64 = _neuralgrok_fp64_params(p_before, grad, opt_obj)

# ── Locate the offending coordinate = argmax|kernel - eager| (the gate's metric).
d_ke = (p_after_kernel.double() - p_after_eager.double()).abs()
i = int(torch.argmax(d_ke).item())
# Map flat index i -> (param name, local idx).
off = 0; off_name = None; off_local = None
for n_, p in named0:
    k = p.numel()
    if off <= i < off + k:
        off_name = n_; off_local = i - off; break
    off += k

abs_ke = d_ke[i].item()
ref_mag_global = p_after_eager.double().abs().max().item()
rel_gate = abs_ke / (ref_mag_global + 1e-30)
print(f"\n[(1a) gate metric] max|Δp(kernel,eager)| = {abs_ke:.3e}  "
      f"rel = {rel_gate:.3e}  (tol 1e-4 -> {'FAIL' if rel_gate>1e-4 else 'PASS'})")
print(f"[offending coord]  name={off_name!r}  local_idx={off_local}  "
      f"flat_idx={i}")
print(f"                   |p_ref| at that coord = {abs(p_after_eager[i].item()):.4f}  "
      f"(reference magnitude there)")

# Per-coordinate breakdown at the worst index: kernel vs eager vs fp64 oracle.
pk = p_after_kernel[i].item(); pe = p_after_eager[i].item(); po = p_after_fp64[i].item()
gi = grad[i].item(); gampi = g_amp_fp64[i].item()
print(f"\n[coord {i}] grad={gi:+.6e}  g_amp(fp64,H16)={gampi:+.6e}  "
      f"sign(g_amp)={'+' if gampi>=0 else '-'}")
print(f"           p_before        = {p_before[i].item():+.8f}")
print(f"           p_after KERNEL  = {pk:+.8f}   (Δ vs before {pk-p_before[i].item():+.3e})")
print(f"           p_after EAGER   = {pe:+.8f}   (Δ vs before {pe-p_before[i].item():+.3e})")
print(f"           p_after FP64    = {po:+.8f}   (Δ vs before {po-p_before[i].item():+.3e})  <-- ARBITER (H=16)")

# ── THE CLASSIFIER: score kernel and eager against the fp64 H=16 oracle. ─────────
# bf16 relative ulp ~ 2^-8 ~ 3.9e-3; the reduced grad rounds at bf16 storage points
# in the wgmma backward, so a kernel/eager that AGREES with the fp64-H16 update to
# ~bf16 roundoff is "within bf16 precision". A SIGN FLIP of g_amp (psi*alpha+beta
# crossing zero) makes the AdamW step jump by 2*lr — that is NOT roundoff.
def _score(p_after, tag):
    d = (p_after.double() - p_after_fp64).abs()
    j = int(torch.argmax(d).item())
    rel = d.max().item() / (p_after_fp64.abs().max().item() + 1e-30)
    print(f"  {tag:26s} vs fp64-H16 oracle: max|Δ|={d.max().item():.3e}  "
          f"rel={rel:.3e}  worst_flat_idx={j}")
    return d.max().item(), rel

print("\n[ARBITER scores — both implementations vs the FULL-fp64 H=16 update on the "
      "kernel's own reduced grad]")
k_abs, k_rel = _score(p_after_kernel, "KERNEL (megakernel,H=16)")
e_abs, e_rel = _score(p_after_eager, "EAGER  (per-op,NG_H=64)")

# bf16 floor for an O(|p|) param under this update: |Δp| ~ lr*(1+wd*|p|); a roundoff
# difference is <~ bf16_ulp * that. A 2*lr jump is the sign-flip signature.
lr = float(opt_obj.param_groups[0]["lr"])
two_lr = 2.0 * lr
bf16_ulp = 2.0 ** -8
print(f"\n  2*lr (one-coord sign-flip signature) = {two_lr:.3e}")
print(f"  0.05*lr (legacy zero-grad eps-noise floor) = {0.05*lr:.3e}")

print("\n=== VERDICT ===")
KERNEL_BF16_OK = k_abs <= max(8.0 * bf16_ulp * (1.0 + abs(p_after_fp64).max().item()),
                              0.05 * lr)
if KERNEL_BF16_OK and e_abs > 0.5 * two_lr:
    print("(a)-variant: KERNEL matches the fp64-H16 oracle to within bf16 roundoff; the "
          "EAGER reference diverges (≈2*lr sign-flip). The kernel-under-test is CORRECT. "
          "The (1a) FAIL is the GATE's eager reference computing psi at the WRONG width "
          "(per-op neuralgrok kernel NG_H=64 vs the deployed/megakernel kPsiHidden=16 — "
          "reads 48 floats OOB per weight array; the OOB garbage is seed/allocation-"
          "dependent and at seeds 123/7 flips sign(g_amp) for one coord). FIX is the gate, "
          "NOT the kernel: anchor (1a) to the fp64-H16 oracle (like (1b) already is) and/or "
          "add the standard 0.05*lr abs floor; the H=64 per-op reference must not gate the "
          "H=16 kernel. (The per-op NG_H=64 width is itself a real, separate, out-of-scope "
          "kernel defect.)")
elif not KERNEL_BF16_OK and k_abs >= 0.5 * two_lr:
    print("(b): KERNEL DIVERGES from the fp64-H16 oracle by ~2*lr (a real g_amp sign flip in "
          "the kernel tail). This is a REAL bf16 accuracy gap/bug in the neuralgrok kernel "
          "tail — FIX the kernel, not the tolerance.")
else:
    print(f"INDETERMINATE: kernel rel={k_rel:.2e}, eager rel={e_rel:.2e}. Inspect the per-"
          f"coord block above; compare seed 42 (passes) vs 123/7 (fail) to confirm the "
          f"sign-flip is reference-side.")
