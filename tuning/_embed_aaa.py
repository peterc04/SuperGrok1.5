"""A/A/A bit-identity + simple cross-impl check for the embed-grad hill-climb.
Builds the d-scaled bench variant, runs tc_train_step 3x from identical inputs,
and asserts the returned reduced-grad tensor is BITWISE identical across reruns.
Also prints a hash so before/after kernels can be compared structurally.
"""
import sys, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tuning.decoder_bench import build_variant

def run(d, B, name):
    mod = build_variant(d, profile=False, name=name)
    dev = torch.device("cuda")
    total = int(mod.TOTAL); vocab = int(mod.VOCAB); seq = int(mod.SEQ)
    def fresh():
        g = torch.Generator(device=dev).manual_seed(42)
        params = (torch.randn(total, generator=g, device=dev) * 0.02).contiguous()
        tokens = torch.randint(0, vocab, (B, seq), generator=g, device=dev).to(torch.int32).contiguous()
        targets = torch.randint(0, vocab, (B,), generator=g, device=dev).to(torch.int32).contiguous()
        state = torch.zeros(3 * total, dtype=torch.float32, device=dev)
        return params, tokens, targets, state
    hashes = []
    grads = []
    for _ in range(3):
        params, tokens, targets, state = fresh()
        loss, grad = mod.tc_train_step(params, tokens, targets, state,
                                       1e-3, 0.9, 0.98, 1e-8, 0.0, 1.0-0.9, 1.0-0.98, 1, 0)
        gb = grad.detach().cpu().contiguous().numpy().tobytes()
        hashes.append(hashlib.sha256(gb).hexdigest())
        grads.append(grad.detach().cpu().clone())
    ok = len(set(hashes)) == 1
    print(f"[A/A/A d={d} B={B}] grad sha256 x3: {[h[:16] for h in hashes]}  -> {'BIT-IDENTICAL' if ok else 'DIVERGED!!!'}")
    # exact max-abs delta between rerun 0 and 1/2 (should be exactly 0.0)
    for k in (1,2):
        md = (grads[0]-grads[k]).abs().max().item()
        print(f"    max|grad[0]-grad[{k}]| = {md:.3e}")
    return ok, hashes[0]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--B", type=int, default=16384)
    ap.add_argument("--name", type=str, default=None)
    a = ap.parse_args()
    ok, h = run(a.d, a.B, a.name)
    sys.exit(0 if ok else 1)
