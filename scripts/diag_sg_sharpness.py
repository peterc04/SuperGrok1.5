import os, torch
os.environ.setdefault("GATE_SEED", "7")
SEED = int(os.environ["GATE_SEED"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import tests.hw.test_l3tc_tail_gate as G
from grokking_optimizers.dispatch import (canonicalize_model, fused_train_step,
                                           _opt_scalars_from)
from tests.hw.test_mamba_tc import _bf16_faithful_mamba_oracle, _pure_fp64_oracle


def classify(cell_key):
    spec = G._CELLS[cell_key]
    model, opt = spec["model"], spec["opt"]
    print(f"\n==================== {cell_key}  (GATE_SEED={SEED}) ====================")
    g, c, m, data, dev = G._build_cell(model)
    canon = canonicalize_model(model)
    tx, ty = data[0], data[1]
    opt_obj = spec["factory"](g, m.parameters(), c)
    named0 = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in named0)
    cache = {}
    loss, grad = fused_train_step(canon, opt, m, opt_obj, tx, ty,
                                  state_cache=cache, step=1, return_grad=True,
                                  gemm_impl="wgmma")
    kstate = cache[canon]["state"]
    k_sharp = kstate[3*total+1: 4*total+1].clone()
    scalars = _opt_scalars_from(opt_obj, 1)
    rho = float(scalars.get("rho", 0.05))
    sam_on = float(scalars.get("looksam_sam", 0.0))
    print(f"  looksam_sam (SAM-step gate @ step1) = {sam_on}   rho = {rho}")
    assert sam_on != 0.0, "SAM 2nd backward did NOT fire at step 1 -- diagnostic would be vacuous"
    g_r, c_r, m_ref, data_r, dev_r = G._build_cell(model)
    named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
    sharp_bf = G._sg_bf16_sharpness_oracle(model, named_ref, tx, ty, grad, rho, total, dev)
    B = int(tx.shape[0]); B16 = B - (B % 16)
    a0 = tx[:B16].detach().cpu().to(torch.long); a1 = ty[:B16].detach().cpu().to(torch.long)
    named_d = {n: p.detach().cpu().double() for n, p in named_ref}
    _lo, go64 = _pure_fp64_oracle(named_d, a0, a1)
    gn = torch.sqrt(sum((v.double()**2).sum() for v in go64.values())).item()
    ron = rho / gn
    off = 0; named_pert = {}
    for n_, p in named_ref:
        k = p.numel()
        named_pert[n_] = (p.detach().cpu().double()
                          + ron * grad[off:off+k].reshape(p.shape).cpu().double())
        off += k
    _lo2, go64p = _pure_fp64_oracle(named_pert, a0, a1)
    sharp_fp = torch.cat([((go64p[n_] - go64[n_]).reshape(-1).double())**2
                          for n_, _ in named_ref]).float().to(dev)

    def relmax(a, b):
        return (a-b).abs().max().item() / (b.abs().max().item() + 1e-30)
    print(f"  REF sharpness max|.|  : bf16-faithful = {sharp_bf.abs().max().item():.4e}   pure-fp64 = {sharp_fp.abs().max().item():.4e}")
    print(f"  kernel sharpness max|.|= {k_sharp.abs().max().item():.4e}")
    sh_abs = (k_sharp - sharp_bf).abs().max().item()
    print(f"  [GATE]   abs d (kernel - bf16faithful) = {sh_abs:.4e}   sharp_rel = {relmax(k_sharp, sharp_bf):.4e}   (tol 2e-2)")
    r_k_fp = relmax(k_sharp, sharp_fp)
    r_bf_fp = relmax(sharp_bf, sharp_fp)
    print(f"  [CLASS]  kernel vs pure-fp64       = {r_k_fp:.4e}")
    print(f"  [CLASS]  bf16faithful vs pure-fp64 = {r_bf_fp:.4e}   (== design bf16 FLOOR on sharpness^2)")
    print(f"  [CLASS]  (kernel-vs-fp64)/(floor)  = {r_k_fp/(r_bf_fp+1e-30):.3f}   --> ~1 => (a) ON floor/BENIGN ; >>1 => (b) REAL gap")
    off = 0; perlayer = {}
    for n_, p in named_ref:
        k = p.numel()
        r = relmax(k_sharp[off:off+k], sharp_fp[off:off+k])
        lk = n_.split(".")[1] if n_.startswith("layers.") else n_
        perlayer[lk] = max(perlayer.get(lk, 0.0), r); off += k
    show = {k: round(v, 4) for k, v in perlayer.items() if k in ("0", "1")}
    print(f"  [STRUCT] per-layer worst kernel-vs-fp64 rel (layer0 vs layer1): {show}  (layer0 >> layer1 => real compounding bug)")


for cell in ("supergrok11/mamba", "supergrok15/mamba"):
    classify(cell)
print("\nDONE.")
