#!/usr/bin/env python3
"""FLAGSHIP 1.5B decoder training, ONE optimizer per process. Builds the scratchpad
OptId-generic STAGED TU (mega_decoder_staged_tc.cu) against the flagship layout
WITHOUT -DSG_DEC_BENCH_LAYOUT=1 (so kDecStagedOptScratch=TRUE → the 4 staged-opt
scratch regions are carved), then runs N steps on a FIXED batch (overfit) banking
loss every LOG_EVERY steps. Handles ALL 11 opts: the 5 elementwise + Prodigy/Muon/
LookSAM/SG11/SG15 via tc_train_step_opt, and SuperGrok2 via sg2_train_step (with the
26-pack meta-net weight bundle from CSAHCAMetaNet.get_weights). SG2 runs at ncta=1.

Env:
  SG_OPT   = adamw|lion|grokfast|grokadamw|neuralgrok|prodigy|muon|looksam|
             supergrok11|supergrok15|supergrok2   (default adamw)
  SG_SEED  = int param-init seed (default 0)
  SG_STEPS = int (default 100)
  SG_LR    = float (default per-opt)
  SG_NCTA  = int ncta_cap (default 8; SG2 forced to 1)
  SG_OUT   = output json path
  SG_TAG   = label for logging
"""
import os, sys, math, json, time
REPO = "/workspace/SuperGrok1.5"; sys.path.insert(0, REPO); os.chdir(REPO)
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
import torch
from torch.utils.cpp_extension import load

VOCAB, SEQ = 99, 4
CUTLASS = [REPO + "/third_party/cutlass/include", REPO + "/third_party/cutlass/tools/util/include"]
TU = os.environ.get("SG_TU",
    "/tmp/claude-0/-/6354dc07-b50f-40a0-8748-5189102539d3/scratchpad/mega_decoder_staged_tc.cu")

OPT_IDS = {"adamw": 0, "lion": 1, "grokfast": 2, "grokadamw": 3, "looksam": 4,
           "prodigy": 5, "neuralgrok": 6, "muon": 7, "supergrok11": 8,
           "supergrok15": 9, "supergrok2": 10}
DEFAULT_LR = {"adamw": 3e-3, "lion": 3e-4, "grokfast": 2e-3, "grokadamw": 2e-3,
              "neuralgrok": 2e-3, "looksam": 3e-3, "prodigy": 1.0, "muon": 3e-3,
              "supergrok11": 3e-3, "supergrok15": 3e-3, "supergrok2": 1e-3}


def build():
    # Per-variant build dir (SG2 rebuilds the TU after the slow-plane alias change;
    # use a distinct dir so it can't clobber the cached .so the running 10-opt batch
    # already loaded). SG_BUILD_TAG selects it.
    tag = os.environ.get("SG_BUILD_TAG", "")
    name = "mega_decoder_staged" + (("_" + tag) if tag else "")
    bd = "/workspace/flagship_build/" + name; os.makedirs(bd, exist_ok=True)
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda",
             "-gencode=arch=compute_90a,code=sm_90a", "-gencode=arch=compute_90a,code=compute_90a",
             "-lineinfo", "-DWITH_CUTLASS", "-DSG_TUNED_GEMM_IMPL=1", "-DSG_DEC_SCALAR_MEGAKERNEL=0",
             "-DSG_FUSED_SM90_DECODER_LAYOUT_CUH_=1",
             "-include", "csrc/fused/sm_90/decoder_flagship_layout.cuh"]
    return load(name=name, sources=[TU], extra_include_paths=[REPO] + CUTLASS,
                extra_cuda_cflags=flags, extra_cflags=["-O3", "-std=c++17"],
                build_directory=bd, verbose=True)


def build_sg2_bundle(dev, seed):
    """Build the model-independent SG2 meta-net weight bundle from CSAHCAMetaNet.
    Returns a dict of contiguous fp32 cuda tensors in the sg2_train_step ABI order."""
    from grokking_optimizers.optimizers.supergrok2 import CSAHCAMetaNet
    torch.manual_seed(seed)
    net = CSAHCAMetaNet(
        d_model=8, num_peer_heads=4, peer_topk=4, num_experts=144,
        expert_hidden=16, gru_hidden=4, n_heads=2, csa_compress=4,
        csa_window=8, csa_topk=16, hca_compress=128, indexer_rank=4,
        rescale=0.1, grad_checkpoint=False).float().to(dev).eval()
    w = net.get_weights(None)
    ne = net.num_experts
    peer_Ws = torch.stack([q.float().contiguous() for q in w['peer_queries']]).contiguous()
    pkA = torch.stack([k.float().contiguous() for k in w['product_keys_A']]).contiguous()
    pkB = torch.stack([k.float().contiguous() for k in w['product_keys_B']]).contiguous()
    b = dict(
        input_proj_W=w['input_proj_W'], input_proj_b=w['input_proj_b'],
        csa_q_W=w['csa_q_W'], csa_k_W=w['csa_k_W'], csa_v_W=w['csa_v_W'], csa_out_W=w['csa_out_W'],
        csa_compress_w=w['csa_compress_w'], csa_idx_DQ=w['csa_idx_DQ'], csa_idx_K=w['csa_idx_K'],
        hca_q_W=w['hca_q_W'], hca_k_W=w['hca_k_W'], hca_v_W=w['hca_v_W'], hca_out_W=w['hca_out_W'],
        gru_Wz=w['gru_W_z'], gru_bz=w['gru_b_z'], gru_Wr=w['gru_W_r'], gru_br=w['gru_b_r'],
        gru_Wh=w['gru_W_h'], gru_bh=w['gru_b_h'],
        peer_query_Ws=peer_Ws, prod_keys_A=pkA, prod_keys_B=pkB,
        expert_W1=w['expert_W1'].reshape(ne, -1).contiguous(), expert_b1=w['expert_b1'].contiguous(),
        expert_W2=w['expert_W2'].reshape(ne, -1).contiguous(), expert_b2=w['expert_b2'].reshape(-1).contiguous())
    return {k: v.float().contiguous().to(dev) for k, v in b.items()}


def main():
    opt = os.environ.get("SG_OPT", "adamw").lower()
    seed = int(os.environ.get("SG_SEED", "0"))
    steps = int(os.environ.get("SG_STEPS", "100"))
    lr = float(os.environ.get("SG_LR", DEFAULT_LR.get(opt, 2e-3)))
    ncta = int(os.environ.get("SG_NCTA", "8"))
    out = os.environ.get("SG_OUT", f"/tmp/claude-0/-/6354dc07-b50f-40a0-8748-5189102539d3/scratchpad/staged_{opt}_s{seed}.json")
    tag = os.environ.get("SG_TAG", f"{opt}/seed{seed}")
    log_every = 10
    assert opt in OPT_IDS, f"unknown opt {opt}; known={list(OPT_IDS)}"
    opt_id = OPT_IDS[opt]
    if opt == "supergrok2":
        ncta = 1  # SG2 per-CTA meta-net workspace is O(Nmax); full occupancy OOMs.

    t0 = time.time()
    mod = build()
    build_s = time.time() - t0
    total = int(mod.TOTAL); NT = int(mod.NUM_TENSORS); GH = int(mod.SG2_GRU_HIDDEN)
    phi_pack = int(mod.SG_PHI_PACK_FLOATS)
    dev = "cuda"; B = 16
    print(f"[{tag}] params={total:,} D={mod.D} L={mod.LAYERS} NT={NT} GH={GH} | "
          f"opt={opt}(id={opt_id}) lr={lr} steps={steps} ncta={ncta} seed={seed} | build={build_s:.1f}s",
          flush=True)

    g = torch.Generator(device=dev).manual_seed(seed)
    params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()

    # State sizing: SG2 = (4+1+GH)*total+1; SG11/15 = 4*total+1+total+phi_pack;
    # Prodigy = 4*total+1+total+3; generic = 4*total+1+total+phi_pack covers all non-SG2.
    if opt == "supergrok2":
        # `slow` plane is aliased onto the dead LookSAM workspace region in the TU,
        # so SG2 state is one plane smaller: [m|v|mu|loss|sharpness|gru_state].
        state_floats = (3 + 1 + GH) * total + 1
    else:
        state_floats = 4 * total + 1 + total + phi_pack
    state = torch.zeros(state_floats, dtype=torch.float32, device=dev)

    # NeuralGrok psi pack seed (extra slice head); SG11/15 phi pack seed (after sharpness).
    if opt == "neuralgrok":
        gp = torch.Generator(device=dev).manual_seed(seed + 100)
        psi = (torch.randn(3 * 16 + 1, generator=gp, device=dev) * 0.1)
        state[2 * total: 2 * total + (3 * 16 + 1)] = psi
    if opt in ("supergrok11", "supergrok15"):
        gp = torch.Generator(device=dev).manual_seed(seed + 200)
        phi = (torch.randn(phi_pack, generator=gp, device=dev) * 0.1)
        phi_base = 3 * total + 1 + total  # loss+1 = sharpness base; +total = phi base
        state[phi_base: phi_base + phi_pack] = phi
    # Prodigy anchor p0 (param_init = state+3*total+1) seeded = params at step 1.
    if opt == "prodigy":
        state[3 * total + 1: 3 * total + 1 + total] = params

    bundle = build_sg2_bundle(dev, seed + 300) if opt == "supergrok2" else None
    if opt == "supergrok2":
        import gc; gc.collect(); torch.cuda.empty_cache()

    gt = torch.Generator(device=dev).manual_seed(7)  # SAME fixed batch across opts.
    tokens = torch.randint(0, VOCAB, (B, SEQ), generator=gt, device=dev, dtype=torch.int32)
    targets = torch.randint(0, VOCAB, (B,), generator=gt, device=dev, dtype=torch.int32)

    b1, b2 = 0.9, 0.98
    # per-tensor scalar arrays for SG2 (length NT). gru_decay==beta1.
    if opt == "supergrok2":
        sc_alpha = torch.full((NT,), 0.96, dtype=torch.float32, device=dev)
        sc_beta1 = torch.full((NT,), b1, dtype=torch.float32, device=dev)
        sc_gru_decay = sc_beta1.clone()
        sc_lamb_eff = torch.full((NT,), 2.0, dtype=torch.float32, device=dev)

    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[{tag}] before-loop GPU mem: free={free_b/1e9:.2f}GB total={total_b/1e9:.2f}GB "
          f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    losses, logged = [], []
    ts = time.time()
    for step in range(1, steps + 1):
        bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
        if opt == "supergrok2":
            sc_bc1 = torch.full((NT,), bc1, dtype=torch.float32, device=dev)
            sc_bc2 = torch.full((NT,), bc2, dtype=torch.float32, device=dev)
            sam_on = 1.0 if (step % 5 == 1) else 0.0  # every-5 SAM-step cadence
            if step == 1:
                fb, tb = torch.cuda.mem_get_info()
                print(f"[{tag}] step1 pre-launch: free={fb/1e9:.2f}GB alloc={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
            ret = mod.sg2_train_step(
                params, tokens, targets, state,
                bundle['input_proj_W'], bundle['input_proj_b'],
                bundle['csa_q_W'], bundle['csa_k_W'], bundle['csa_v_W'], bundle['csa_out_W'],
                bundle['csa_compress_w'], bundle['csa_idx_DQ'], bundle['csa_idx_K'],
                bundle['hca_q_W'], bundle['hca_k_W'], bundle['hca_v_W'], bundle['hca_out_W'],
                bundle['gru_Wz'], bundle['gru_bz'], bundle['gru_Wr'], bundle['gru_br'],
                bundle['gru_Wh'], bundle['gru_bh'],
                bundle['peer_query_Ws'], bundle['prod_keys_A'], bundle['prod_keys_B'],
                bundle['expert_W1'], bundle['expert_b1'], bundle['expert_W2'], bundle['expert_b2'],
                sc_alpha, sc_gru_decay, sc_lamb_eff, sc_beta1, sc_bc1, sc_bc2,
                0.1, b2, lr, 0.01, 1e-8, 0.05, sam_on, step, 1)
            loss = ret[0]
        else:
            # SAM cadence for LookSAM/SG11/SG15; inert (0) for non-SAM opts.
            looksam_sam = 1.0 if (opt in ("looksam", "supergrok11", "supergrok15") and step % 5 == 1) else 0.0
            rho = 0.05 if opt in ("looksam", "supergrok11", "supergrok15") else 0.0
            sg_rescale = 0.1 if opt in ("supergrok11", "supergrok15") else 0.0
            beta3 = math.sqrt(b2) if opt == "prodigy" else 0.0
            d0 = float(os.environ.get("SG_D0", "1e-6"))  # Prodigy cold-start d_prev
            loss, _ = mod.tc_train_step_opt(
                opt_id, params, tokens, targets, state, lr, b1, b2, 1e-8, 0.0,
                bc1, bc2, step, 0.98, 2.0, 0.0,
                rho, looksam_sam, sg_rescale, 0.5, 1.0, d0, 1.0, beta3, ncta)
        l = float(loss.item()); losses.append(l)
        if not math.isfinite(l):
            print(f"[{tag}] step {step} NON-FINITE loss={l} -> abort", flush=True)
            break
        if step % log_every == 0 or step <= 3 or step == steps:
            logged.append([step, l])
            print(f"[{tag}]  step {step:4d}  loss={l:.5f}", flush=True)
    dur = time.time() - ts
    finite = all(math.isfinite(x) for x in losses)
    decreased = (losses[-1] < losses[0] - 0.5) if losses else False
    res = {"tag": tag, "opt": opt, "opt_id": opt_id, "seed": seed, "lr": lr, "ncta": ncta,
           "steps_run": len(losses), "params": total, "D": int(mod.D), "L": int(mod.LAYERS),
           "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
           "decreased": decreased, "finite": finite, "logged": logged,
           "all_losses": losses, "step_time_s": dur / max(len(losses), 1), "build_s": build_s,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?")}
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"[{tag}] DONE loss {res['loss_first']:.4f} -> {res['loss_last']:.4f} "
          f"(decreased? {decreased}; finite? {finite}) {res['step_time_s']*1000:.0f}ms/step -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
