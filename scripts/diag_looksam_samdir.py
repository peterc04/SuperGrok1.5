import os, torch
os.environ.setdefault("GATE_SEED", "7")
SEED = int(os.environ["GATE_SEED"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import tests.hw.test_l3tc_tail_gate as G
from grokking_optimizers.dispatch import (canonicalize_model, fused_train_step,
                                           _opt_scalars_from)
from tests.hw.test_mamba_tc import _bf16_faithful_mamba_oracle, _pure_fp64_oracle

cell_key = "looksam/mamba"
spec = G._CELLS[cell_key]; model, opt = spec["model"], spec["opt"]
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
k_sam = kstate[2*total:3*total].clone()
rho = float(opt_obj.param_groups[0]["rho"])
kk = int(opt_obj.param_groups[0].get("k", 5))
sam_on = 1.0 if ((1-1) % max(kk, 1) == 0) else 0.0
print(f"  k={kk}  rho={rho}  SAM-step@1 = {sam_on}  (must be 1)")
assert sam_on != 0.0
g_r, c_r, m_ref, data_r, dev_r = G._build_cell(model)
named_ref = [(n, p) for n, p in m_ref.named_parameters() if p.requires_grad]
B = int(tx.shape[0]); B16 = B - (B % 16)
a0 = tx[:B16].detach().cpu().to(torch.long); a1 = ty[:B16].detach().cpu().to(torch.long)
named_bf = {n: p.detach().cpu().double() for n, p in named_ref}
_lo, go_bf = _bf16_faithful_mamba_oracle(named_bf, a0, a1)
gn_bf = torch.sqrt(sum((v.double()**2).sum() for v in go_bf.values())).item()
ron_bf = rho / gn_bf
off = 0; pert_bf = {}
for n_, p in named_ref:
    k = p.numel(); pert_bf[n_] = p.detach().cpu().double()+ron_bf*grad[off:off+k].reshape(p.shape).cpu().double(); off += k
_lo2, go_bf2 = _bf16_faithful_mamba_oracle(pert_bf, a0, a1)
sam_bf = torch.cat([(go_bf2[n_]-go_bf[n_]).reshape(-1).double() for n_, _ in named_ref]).float().to(dev)
_lo, go_fp = _pure_fp64_oracle(named_bf, a0, a1)
gn_fp = torch.sqrt(sum((v.double()**2).sum() for v in go_fp.values())).item()
ron_fp = rho / gn_fp
off = 0; pert_fp = {}
for n_, p in named_ref:
    k = p.numel(); pert_fp[n_] = p.detach().cpu().double()+ron_fp*grad[off:off+k].reshape(p.shape).cpu().double(); off += k
_lo2, go_fp2 = _pure_fp64_oracle(pert_fp, a0, a1)
sam_fp = torch.cat([(go_fp2[n_]-go_fp[n_]).reshape(-1).double() for n_, _ in named_ref]).float().to(dev)


def relmax(a, b):
    return (a-b).abs().max().item()/(b.abs().max().item()+1e-30)
print(f"  REF sam_dir max|.| : bf16-faithful={sam_bf.abs().max().item():.4e}  pure-fp64={sam_fp.abs().max().item():.4e}")
print(f"  kernel sam_dir max|.| = {k_sam.abs().max().item():.4e}")
print(f"  [GATE]   abs d (kernel - bf16faithful) = {(k_sam-sam_bf).abs().max().item():.4e}   sam_dir_rel = {relmax(k_sam, sam_bf):.4e}   (tol 2e-2)  <-- looksam/mamba failing assertion")
r_k_fp = relmax(k_sam, sam_fp); r_bf_fp = relmax(sam_bf, sam_fp)
print(f"  [CLASS]  kernel vs pure-fp64       = {r_k_fp:.4e}")
print(f"  [CLASS]  bf16faithful vs pure-fp64 = {r_bf_fp:.4e}   (== bf16 FLOOR on LINEAR g_sam-g)")
print(f"  [CLASS]  (kernel-vs-fp64)/(floor)  = {r_k_fp/(r_bf_fp+1e-30):.3f}   ~1 => (a) on floor/benign ; >>1 => (b) real gap")
print("\nDONE.")
