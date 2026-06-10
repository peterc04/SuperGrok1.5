#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  11-Optimizer Grokking Race — GCP VM Edition (Multi-GPU)
═══════════════════════════════════════════════════════════════════════

3 architectures × 4 splits, all algorithmic tasks:
  Decoder Transformer  →  (a ÷ b) mod 97              [4-token seq]
  Vision Transformer   →  MNIST (a + b) mod 97         [16 patches + CLS]
  Mamba SSM            →  (a÷b₁÷b₂÷b₃) mod 97        [8-token chain]

11 optimizers (all C++/CUDA accelerated via grokking_optimizers package):
  AdamW, NeuralGrok, GrokAdamW, SuperGrok (v1.1), SuperGrok1.5,
  SuperGrok2 (Sparse Attention), Grokfast, Muon, Lion, LookSAM, Prodigy

USAGE:
  # First time — install everything:
  python grokking_race.py --setup

  # Run on single GPU (default — fair sequential benchmark):
  python grokking_race.py

  # Run on multiple GPUs (each optimizer gets its own GPU):
  python grokking_race.py --gpus 0,1,2,3

  # With phone notifications (free, no signup):
  #   1. Install ntfy app on phone: https://ntfy.sh
  #   2. Subscribe to your topic (e.g. "my-grok-run")
  python grokking_race.py --ntfy my-grok-run

  # Query progress from anywhere:
  #   curl http://<VM_EXTERNAL_IP>:8080/status
  #   The status server starts automatically.

  # Text it for a progress report (from phone via ntfy):
  #   Publish "status" to your ntfy topic → it replies with progress.
═══════════════════════════════════════════════════════════════════════
"""

import subprocess, os, sys, argparse

# ─────────────────────────────────────────────────────────────────────
#  PART 0: SETUP (run once with --setup flag)
# ─────────────────────────────────────────────────────────────────────
def run_setup():
    def _sh(cmd):
        print(f"  $ {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    print("=" * 55)
    print("  INSTALLING DEPENDENCIES")
    print("=" * 55)

    _sh("pip install -q torch torchvision matplotlib numpy tqdm requests")

    # Build grokking_optimizers C++/CUDA extension (all optimizers)
    opt_pkg = os.path.join(os.path.dirname(__file__), "grokking_optimizers")
    if not os.path.exists(opt_pkg):
        opt_pkg = "grokking_optimizers"
    if os.path.exists(opt_pkg):
        print("Building grokking_optimizers C++/CUDA extension …")
        _sh(f"pip install -q -e {opt_pkg}/")
    else:
        print("  ⚠ grokking_optimizers/ not found — install with: pip install -e grokking_optimizers/")

    # Pre-download MNIST
    print("Downloading MNIST …")
    import torchvision
    torchvision.datasets.MNIST(root='./data', train=True, download=True)
    print("✓ MNIST cached")

    print("\n" + "=" * 55)
    print("  SETUP COMPLETE — now run without --setup")
    print("=" * 55)
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────
#  PART 0.5: NOTIFICATION + STATUS SERVER
# ─────────────────────────────────────────────────────────────────────
import threading, json, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
import multiprocessing as mp
from multiprocessing import Queue as MPQueue

# Global progress tracker
_PROGRESS = {
    "status": "initializing",
    "mode": None,
    "started_at": None,
    "current_run": 0,
    "total_runs": 0,
    "current_task": "",
    "completed": [],       # list of {"name", "model", "split", "seed", "grokked", "wall_time"}
    "errors": [],
    "eta_seconds": None,
}
_PROGRESS_LOCK = threading.Lock()

def _update_progress(**kw):
    with _PROGRESS_LOCK:
        _PROGRESS.update(kw)

def _progress_snapshot():
    with _PROGRESS_LOCK:
        return dict(_PROGRESS)

class _StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        snap = _progress_snapshot()
        # Compute ETA
        if snap["started_at"] and snap["current_run"] > 0 and snap["total_runs"] > 0:
            elapsed = time.time() - snap["started_at"]
            per_run = elapsed / snap["current_run"]
            remaining = (snap["total_runs"] - snap["current_run"]) * per_run
            snap["eta_seconds"] = round(remaining)
            snap["eta_human"] = str(timedelta(seconds=int(remaining)))
            snap["elapsed_human"] = str(timedelta(seconds=int(elapsed)))
        # Completion stats
        done = snap["completed"]
        grokked = sum(1 for d in done if d.get("grokked"))
        snap["summary"] = f"{len(done)}/{snap['total_runs']} runs done, {grokked} grokked, {len(snap['errors'])} errors"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(snap, indent=2, default=str).encode())

    def log_message(self, format, *args):
        pass  # suppress request logging

def start_status_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), _StatusHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"✓ Status server running on port {port}")
    print(f"  curl http://localhost:{port}/status")
    print(f"  curl http://<VM_EXTERNAL_IP>:{port}/status")
    return server

# ── ntfy.sh notifications ─────────────────────────────────────────────
_NTFY_TOPIC = None

def _ntfy(message, title=None, priority="default", tags=None):
    """Send push notification via ntfy.sh (free, no signup)."""
    if not _NTFY_TOPIC:
        return
    try:
        import requests
        headers = {}
        if title: headers["Title"] = title
        if priority != "default": headers["Priority"] = priority
        if tags: headers["Tags"] = tags
        requests.post(f"https://ntfy.sh/{_NTFY_TOPIC}",
                      data=message.encode(), headers=headers, timeout=5)
    except ImportError:
        print("  ⚠ ntfy: 'requests' library not installed — run: pip install requests")
    except Exception as e:
        print(f"  ⚠ ntfy failed: {e}")

def _start_ntfy_listener():
    """Listen for incoming messages on the ntfy topic and reply with status."""
    if not _NTFY_TOPIC:
        return
    def _listen():
        import requests
        while True:
            try:
                r = requests.get(f"https://ntfy.sh/{_NTFY_TOPIC}/json",
                                 stream=True, timeout=600)
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        if msg.get("event") != "message":
                            continue
                        text = msg.get("message", "").strip().lower()
                        if text in ("status", "progress", "update", "?", "eta"):
                            snap = _progress_snapshot()
                            done = snap["completed"]
                            grokked = sum(1 for d in done if d.get("grokked"))
                            elapsed = time.time() - snap["started_at"] if snap["started_at"] else 0
                            eta_s = ""
                            if snap["current_run"] > 0 and snap["total_runs"] > 0:
                                per_run = elapsed / snap["current_run"]
                                remaining = (snap["total_runs"] - snap["current_run"]) * per_run
                                eta_s = f"\nETA: {timedelta(seconds=int(remaining))}"
                            reply = (
                                f"📊 {len(done)}/{snap['total_runs']} runs complete\n"
                                f"✓ {grokked} grokked | ✗ {len(done)-grokked} DNF | ⚠ {len(snap['errors'])} errors\n"
                                f"⏱ Elapsed: {timedelta(seconds=int(elapsed))}{eta_s}\n"
                                f"🔄 Current: {snap['current_task']}"
                            )
                            _ntfy(reply, title="Progress Report", tags="bar_chart")
                    except Exception:
                        pass
            except Exception:
                time.sleep(10)  # reconnect on failure

    t = threading.Thread(target=_listen, daemon=True)
    t.start()
    print(f"✓ ntfy listener active — text 'status' to ntfy.sh/{_NTFY_TOPIC} for progress")


# ─────────────────────────────────────────────────────────────────────
#  PART 1: SHARED (models + data generators + eval)
# ─────────────────────────────────────────────────────────────────────
import math, copy, random
from typing import Dict, Optional
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

# Gradient checkpointing (activation recomputation) is enabled for all models
# by default — it trades a recompute in backward for a large activation-memory
# saving, which matters for the deep/long-context grokking runs. It is applied
# per transformer/SSM block and is automatically bypassed in eval (no grad) so
# validation/inference cost is unchanged. Override per-run via the "grad_checkpoint"
# config key.
def _maybe_checkpoint(module, x, enabled):
    """Run `module(x)`, optionally under activation checkpointing.

    Only checkpoints when training AND grad is being tracked — under
    torch.no_grad()/eval there is nothing to recompute so we call directly,
    avoiding checkpoint's overhead and its "no input requires grad" warning.
    """
    if enabled and module.training and torch.is_grad_enabled():
        return _grad_checkpoint(module, x, use_reentrant=False)
    return module(x)

MODEL_SCALES = {
    "small":  {"dim_model": 128, "num_heads": 4, "num_layers": 2},   # ~420K params
    "medium": {"dim_model": 256, "num_heads": 8, "num_layers": 4},   # ~3.5M params
    "large":  {"dim_model": 512, "num_heads": 8, "num_layers": 6},   # ~20M params
}

DEFAULT_CONFIG: Dict = {
    "p": 97, "operation": "x/y", "frac_train": 0.5, "val_ratio": 0.10,
    "num_layers": 2, "dim_model": 128, "num_heads": 4, "num_tokens": 99,
    "lr": 1e-3, "weight_decay": 1.0, "beta1": 0.9, "beta2": 0.98,
    "max_steps": 100_000, "early_stop_threshold": 0.95,
    "early_stop_max_steps": 20_000, "eval_every": 10,  # owner: track metrics every 10 gradient steps
    # Owner (2026-06-10): the race stops on VAL grokking (val>=0.95, patience-
    # held); TEST stays the report metric (grokking_step_test_confirmed). No
    # heuristic dead-stop: non-grokking cells run to early_stop_max_steps (an
    # honest DNF) under live supervision.
    "early_stop_on": "val",
    # Matmul precision policy (owner: BF16 tensor-core is the intended default;
    # configurable axis). "bf16" -> autocast(bfloat16) fwd/bwd on tensor cores
    # (fp32 master weights + fp32 grads at the optimizer boundary, so every
    # fused optimizer kernel ABI is unchanged); "tf32" -> fp32 autocast OFF +
    # allow_tf32 GEMMs; "fp32" -> strict CUDA-core fp32 (the legacy baseline).
    # Meta/SAM side-steps and evaluate() deliberately stay fp32 (sharpness and
    # accuracy metrics are numerics-sensitive). Roofline ceilings follow this
    # axis (tuning/roofline.py).
    # Owner decision (2026-06-10): precision program dropped — industry
    # standard fixed everywhere = bf16 mixed precision (bf16 compute, fp32
    # master weights/optimizer state). "auto"/per-model modes remain available
    # as explicit overrides only.
    "matmul_precision": "bf16",
    # [A4-H1] patience counts EVALS, not steps: 50 evals × eval_every=10 =
    # 500-step post-grok hold. The old 500 (× eval_every=100 = 50k steps)
    # EXCEEDED max_steps, so the {metric}_threshold stop was unreachable and
    # every race run burned the full step budget instead of stopping at grok.
    "early_stop_patience": 50, "seed": 42,
    "compile_model": False, "use_amp": False, "model_type": "decoder",
    "patch_dim": 49, "num_patches": 16,
    "chain_length": 3, "seq_len": 8,
    "use_fused": True,
}

# ── Data 1: Modular Division (a ÷ b) mod p  [Decoder] ────────────────
def make_data(p=97, frac_train=0.5, val_ratio=0.10, seed=42):
    rng = random.Random(seed)
    op_tok, eq_tok = p, p + 1
    pairs, labels = [], []
    for a in range(p):
        for b in range(1, p):
            b_inv = pow(b, p - 2, p)
            pairs.append([a, op_tok, b, eq_tok])
            labels.append((a * b_inv) % p)
    c = list(zip(pairs, labels)); rng.shuffle(c)
    pairs, labels = zip(*c)
    n_train_total = int(len(pairs) * frac_train)
    n_val = int(n_train_total * val_ratio)
    n_train = n_train_total - n_val
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x[:n_train], y[:n_train], x[n_train:n_train_total], y[n_train:n_train_total], x[n_train_total:], y[n_train_total:]

# ── Data 2: MNIST-Addition (a + b) mod p  [ViT] ──────────────────────
def make_mnist_addition_data(p=97, frac_train=0.5, val_ratio=0.10, seed=42):
    import torchvision
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((14, 14)), transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    digit_images = {d: [] for d in range(10)}
    for img, label in mnist:
        digit_images[label].append(img.squeeze(0))
    rng = random.Random(seed)
    number_images = {}
    for n in range(p):
        tens, ones = n // 10, n % 10
        img_t = digit_images[tens][rng.randint(0, len(digit_images[tens]) - 1)]
        img_o = digit_images[ones][rng.randint(0, len(digit_images[ones]) - 1)]
        number_images[n] = torch.cat([img_t, img_o], dim=1)
    pairs, labels = [], []
    for a in range(p):
        for b in range(p):
            pairs.append((a, b)); labels.append((a + b) % p)
    combined = list(zip(pairs, labels)); rng.shuffle(combined)
    pairs, labels = zip(*combined)
    n_train_total = int(len(pairs) * frac_train)
    n_val = int(n_train_total * val_ratio)
    n_train = n_train_total - n_val
    images = []
    for a, b in pairs:
        full = torch.cat([number_images[a], number_images[b]], dim=0)
        patches = full.unfold(0, 7, 7).unfold(1, 7, 7).contiguous().reshape(16, 49)
        images.append(patches)
    x = torch.stack(images); y = torch.tensor(labels, dtype=torch.long)
    return x[:n_train], y[:n_train], x[n_train:n_train_total], y[n_train:n_train_total], x[n_train_total:], y[n_train_total:]

# ── Data 3: Sequential Chained Division  [Mamba] ─────────────────────
def make_sequential_division_data(p=97, chain_length=3, frac_train=0.5, val_ratio=0.10, seed=42):
    rng = random.Random(seed)
    op_tok, eq_tok = p, p + 1
    target_size = p * (p - 1)
    seen = set(); pairs, labels = [], []
    while len(pairs) < target_size:
        a = rng.randint(0, p - 1)
        bs = tuple(rng.randint(1, p - 1) for _ in range(chain_length))
        key = (a, *bs)
        if key in seen: continue
        seen.add(key)
        result = a
        for b in bs: result = (result * pow(b, p - 2, p)) % p
        seq = [a]
        for b in bs: seq.extend([op_tok, b])
        seq.append(eq_tok)
        pairs.append(seq); labels.append(result)
    combined = list(zip(pairs, labels)); rng.shuffle(combined)
    pairs, labels = zip(*combined)
    n_train_total = int(len(pairs) * frac_train)
    n_val = int(n_train_total * val_ratio)
    n_train = n_train_total - n_val
    x = torch.tensor(pairs, dtype=torch.long); y = torch.tensor(labels, dtype=torch.long)
    return x[:n_train], y[:n_train], x[n_train:n_train_total], y[n_train:n_train_total], x[n_train_total:], y[n_train_total:]

def make_data_for_task(c, seed):
    mt = c.get("model_type", "decoder"); ft, p = c.get("frac_train", 0.5), c.get("p", 97)
    vr = c.get("val_ratio", 0.10)
    if mt == "decoder":  return make_data(p, ft, vr, seed)
    elif mt == "vit":    return make_mnist_addition_data(p, ft, vr, seed)
    elif mt == "mamba":  return make_sequential_division_data(p, c.get("chain_length", 3), ft, vr, seed)
    else: raise ValueError(f"Unknown model_type: {mt}")

# ── Model 1: Decoder Transformer ─────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, d, h, seq_len=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.register_buffer('causal_mask', torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1))
    def forward(self, x):
        a, _ = self.attn(x, x, x, attn_mask=self.causal_mask)
        x = self.n1(x + a); return self.n2(x + self.ff(x))

class Transformer(nn.Module):
    def __init__(self, nl=2, d=128, h=4, ntok=99, seq=4, grad_checkpoint=True):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(seq, d)
        self.layers = nn.ModuleList([DecoderBlock(d, h, seq_len=seq) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, ntok)
        self.register_buffer('pos_ids', torch.arange(seq).unsqueeze(0))
        self.grad_checkpoint = grad_checkpoint
    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        for l in self.layers: h = _maybe_checkpoint(l, h, self.grad_checkpoint)
        return self.out(self.norm(h)[:, -1, :])

# ── Model 2: ViT ─────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=0., batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a); return self.n2(x + self.ff(x))

class ViT(nn.Module):
    def __init__(self, p=97, patch_dim=49, num_patches=16, d=128, h=4, nl=2, grad_checkpoint=True):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Embedding(num_patches + 1, d)
        self.layers = nn.ModuleList([EncoderBlock(d, h) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, p)
        self.register_buffer('pos_ids', torch.arange(num_patches + 1).unsqueeze(0))
        self.grad_checkpoint = grad_checkpoint
    def forward(self, x):
        B = x.size(0); h = self.patch_proj(x)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = h + self.pos(self.pos_ids)
        for l in self.layers: h = _maybe_checkpoint(l, h, self.grad_checkpoint)
        return self.out(self.norm(h[:, 0, :]))

# ── Model 3: Mamba SSM ───────────────────────────────────────────────
class SelectiveSSMLayer(nn.Module):
    def __init__(self, d, state_dim=16, dt_rank=None, expand_factor=2):
        super().__init__()
        self.state_dim = state_dim; self.d_inner = d * expand_factor
        self.dt_rank = dt_rank or max(d // 16, 1)
        self.in_proj = nn.Linear(d, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3,
                                padding=1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)
        self.norm = nn.LayerNorm(d)
    def _selective_scan(self, x, dt, B, C):
        batch, L, _ = x.shape; A = -torch.exp(self.A_log); dt = F.softplus(dt)
        # Try CUDA kernel
        if x.is_cuda:
            try:
                from mamba_scan_ext import selective_scan_cuda
                return selective_scan_cuda(
                    x.contiguous(), dt.contiguous(),
                    B.contiguous(), C.contiguous(), A.contiguous()
                )
            except ImportError:
                pass
        # Python fallback
        h = torch.zeros(batch, self.d_inner, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            h = torch.exp(dt_t * A.unsqueeze(0)) * h + (dt_t * B[:, t, :].unsqueeze(1)) * x[:, t, :].unsqueeze(-1)
            ys.append((h * C[:, t, :].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1)
    def forward(self, x):
        residual = x; xz = self.in_proj(x); x_main, z = xz.chunk(2, dim=-1)
        x_main = F.silu(self.conv1d(x_main.transpose(1, 2)).transpose(1, 2))
        x_dbc = self.x_proj(x_main)
        dt, B, C = x_dbc.split([self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        y = self._selective_scan(x_main, self.dt_proj(dt), B, C)
        y = self.out_proj((y + x_main * self.D.unsqueeze(0).unsqueeze(0)) * F.silu(z))
        return self.norm(y + residual)

class MambaModel(nn.Module):
    def __init__(self, p=97, ntok=99, seq_len=8, d=128, nl=2, grad_checkpoint=True):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(seq_len, d)
        self.layers = nn.ModuleList([SelectiveSSMLayer(d) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, p)
        self.register_buffer('pos_ids', torch.arange(seq_len).unsqueeze(0))
        self.grad_checkpoint = grad_checkpoint
    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        for l in self.layers: h = _maybe_checkpoint(l, h, self.grad_checkpoint)
        return self.out(self.norm(h[:, -1, :]))

# ── Model Factory ─────────────────────────────────────────────────────
MODEL_LABELS = {
    "decoder": "Decoder Transformer  [a÷b mod p]",
    "vit":     "ViT  [MNIST-Addition mod p]",
    "mamba":   "Mamba SSM  [Seq. Division mod p]",
}
TASK_LABELS = {"decoder": "(a ÷ b) mod 97", "vit": "MNIST (a + b) mod 97", "mamba": "(a÷b₁÷b₂÷b₃) mod 97"}

def _raw_model(c, device):
    mt, p, d, h, nl = c.get("model_type","decoder"), c["p"], c["dim_model"], c["num_heads"], c["num_layers"]
    gc = c.get("grad_checkpoint", True)
    if mt == "decoder": return Transformer(nl, d, h, c["num_tokens"], 4, grad_checkpoint=gc).to(device)
    elif mt == "vit":   return ViT(p=p, patch_dim=c.get("patch_dim",49), num_patches=c.get("num_patches",16), d=d, h=h, nl=nl, grad_checkpoint=gc).to(device)
    elif mt == "mamba": return MambaModel(p=p, ntok=c["num_tokens"], seq_len=c.get("seq_len",8), d=d, nl=nl, grad_checkpoint=gc).to(device)
    else: raise ValueError(f"Unknown: {mt}")

def build_model(c, device, do_compile=False):
    m = _raw_model(c, device)
    if not do_compile or not hasattr(torch, "compile"): return m
    try:
        cm = torch.compile(m, mode="reduce-overhead", fullgraph=False)
        mt = c.get("model_type","decoder")
        if mt == "vit":     dummy = torch.randn(2, c.get("num_patches",16), c.get("patch_dim",49), device=device)
        elif mt == "mamba": dummy = torch.zeros(2, c.get("seq_len",8), dtype=torch.long, device=device)
        else:               dummy = torch.zeros(2, 4, dtype=torch.long, device=device)
        with torch.no_grad(): _ = cm(dummy)
        return cm
    except Exception: return m

def get_init_state(c, device):
    torch.manual_seed(c["seed"])
    return copy.deepcopy(_raw_model(c, device).state_dict())

@torch.no_grad()
def evaluate(model, x, y, p=97):
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits[:, :p].argmax(-1) == y).float().mean().item()
    return loss, acc

class EarlyStopper:
    def __init__(self, threshold=0.95, max_steps=20_000, patience=500, metric_name="test_acc"):
        self.threshold=threshold; self.max_steps=max_steps; self.patience=patience
        self.metric_name=metric_name  # which accuracy feeds step(): "test_acc" or "val_acc"
        # [A4-M1/M2] best_metric_acc / metric_acc are metric-agnostic names: under
        # val-stopping this tracks val, under test-stopping it tracks test — the
        # old test_acc-specific names mislabelled the val-criterion runs.
        self._triggered=False; self._counter=0; self.best_metric_acc=0.
        self.grokking_step=None; self.grokking_wall=None; self._t0=time.time()
        self.stopping_reason=None; self.stopping_step=None
    def step(self, metric_acc, current_step):
        if current_step >= self.max_steps:
            if self.stopping_reason is None:
                self.stopping_reason="max_steps"; self.stopping_step=current_step
            return True
        self.best_metric_acc = max(self.best_metric_acc, metric_acc)
        if metric_acc >= self.threshold:
            if not self._triggered:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                self._triggered=True; self.grokking_step=current_step
                self.grokking_wall = time.time()-self._t0
            self._counter += 1
            if self._counter >= self.patience:
                if self.stopping_reason is None:
                    self.stopping_reason=f"{self.metric_name}_threshold"; self.stopping_step=current_step
                return True
        else: self._counter=0
        return False


# ─────────────────────────────────────────────────────────────────────
#  PART 2: TRAINING LOOPS
# ─────────────────────────────────────────────────────────────────────
from pathlib import Path
from collections import defaultdict
import numpy as np, warnings
from tqdm.auto import tqdm

REPO = Path(".") / "repos"

# ── JIT-compile Mamba CUDA scan kernel (if available) ──────────────────
try:
    from torch.utils.cpp_extension import load as _load_ext
    _mamba_scan_src = os.path.join(os.path.dirname(__file__), "mamba_scan_kernel.cu")
    if os.path.exists(_mamba_scan_src) and torch.cuda.is_available():
        import mamba_scan_ext  # noqa: F401 — already built
except ImportError:
    try:
        mamba_scan_ext = _load_ext(
            name="mamba_scan_ext",
            sources=[_mamba_scan_src],
            verbose=False,
        )
    except Exception:
        pass  # Fall back to Python scan
except Exception:
    pass

class TrainResult:
    __slots__ = ("name","seed","steps","train_losses","train_accs",
                 "val_losses","val_accs","test_losses","test_accs",
                 "wall_time","total_steps",
                 "grokking_step","grokking_wall","final_val_acc","final_train_acc",
                 "final_test_acc","final_test_loss","final_val_loss",
                 "stopping_reason","stopping_step","val_test_gap",
                 "model_type","frac_train","val_ratio","component_failures",
                 # [A4-M3] TEST-confirmed grok flag (val-trained metas can fake-
                 # grok val); [A4-M2] best metric-criterion acc seen by the stopper
                 "grokking_step_test_confirmed","best_metric_acc",
                 # [A4-M4] per-run resolved matmul precision + AMP flag (each
                 # optimizer can carry a different tuned precision)
                 "matmul_precision","use_amp",
                 # WIRING GUARD (task 2): which execution path the train loop
                 # ACTUALLY took this run — "L3-TC bf16" / "L3-scalar fp32" /
                 # "L1+eager". Set once (first step) by the train_* loop; carried
                 # into the run JSON so a silently-degraded run is visible.
                 "train_path",
                 # Whether fusion was REQUESTED this run (c["use_fused"]). The FLOP
                 # pass disables it deliberately; the guard must not flag that as a
                 # degrade — so _record_train_path reads this fact off the result.
                 "use_fused_requested")
    def __init__(self, name, seed=42, model_type="decoder", frac_train=0.5, val_ratio=0.10,
                 matmul_precision="auto", use_amp=False):
        self.name=name; self.seed=seed; self.model_type=model_type
        self.frac_train=frac_train; self.val_ratio=val_ratio
        self.matmul_precision=matmul_precision; self.use_amp=use_amp
        self.train_path=None  # set once per run by the train_* loop (task 2)
        self.use_fused_requested=True  # overwritten by _tr from c["use_fused"]
        self.steps=[]; self.train_losses=[]; self.train_accs=[]
        self.val_losses=[]; self.val_accs=[]
        self.test_losses=[]; self.test_accs=[]
        self.wall_time=0.; self.total_steps=0; self.grokking_step=None
        self.grokking_wall=None; self.final_val_acc=0.; self.final_train_acc=0.
        self.final_test_acc=0.; self.final_test_loss=0.; self.final_val_loss=0.
        self.stopping_reason=None; self.stopping_step=None; self.val_test_gap=0.
        # per-component failure counts (sam_step/meta_step/bilevel_step) — a
        # component that breaks must be VISIBLE in results, never silent
        self.component_failures={}
        # [A4-M3] True iff grokking_step is set AND the eval nearest it had
        # test_acc >= threshold-0.05 (set in _fin); guards against val-only
        # fake-grok by val-trained meta-nets. [A4-M2] best metric-criterion acc.
        self.grokking_step_test_confirmed=False; self.best_metric_acc=0.

def _merge(base, ov):
    m = dict(base)
    if ov: m.update(ov)
    return m
# Measured per-model precision verdicts (results/h100_grokking_race/
# PRECISION_ANALYSIS.md, 3 seeds/cell): decoder stands at EVERY precision
# (bf16 = fastest that fully stands); vit only fully stands at fp32 (tf32 lost
# 1/3 seeds, bf16 2/3); mamba untested -> conservative fp32 until measured.
# The tuner additionally tunes precision per (optimizer, model); "auto" is the
# default for untuned contexts.
_AUTO_PRECISION = {"decoder": "bf16", "vit": "fp32", "mamba": "fp32"}

def _resolve_matmul_precision(c):
    mp = c.get("matmul_precision", "auto")
    if mp == "auto":
        return _AUTO_PRECISION.get(c.get("model_type", "decoder"), "fp32")
    return mp

def _autocast(c):
    """Forward/backward autocast per the matmul_precision axis.

    Precedence: legacy use_amp=True keeps the old fp16+GradScaler behavior;
    otherwise matmul_precision selects bf16 autocast (default) or none
    (tf32/fp32 — those differ via the allow_tf32 backend flag set in
    _apply_matmul_precision). GradScaler stays enabled only for fp16 (bf16
    needs no loss scaling; the existing scaler calls are no-op passthroughs
    when disabled)."""
    if c.get("use_amp", False):
        return torch.amp.autocast('cuda')
    mp = _resolve_matmul_precision(c)
    if mp == "bf16" or mp in ("fp8", "fp8e5m2", "int8"):
        # fp8/int8 swap the Linear GEMMs to native kernels (lowprec.py); all
        # other ops ride the standard bf16 carrier, as in TE-style recipes.
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()

def _apply_matmul_precision(c):
    tf32 = (_resolve_matmul_precision(c) == "tf32")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

def _stopper(c):
    # early_stop_on: "test" (default, historical) or "val" — which accuracy
    # triggers the threshold/patience stop. Tuner + the val-criterion race use "val".
    metric = "val_acc" if c.get("early_stop_on", "test") == "val" else "test_acc"
    return EarlyStopper(c["early_stop_threshold"], c.get("early_stop_max_steps", c["max_steps"]),
                        c["early_stop_patience"], metric_name=metric)
def _pbar(name, mx, pos):
    return tqdm(range(1, mx+1), desc=f"{name:<14s}", position=pos, leave=True, ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]")
def _progressive_eval_freq(step, base_freq=10, max_freq=50, scale=0.01, thresh=500):
    """Sigmoid-driven eval frequency: eval less often early, more often later."""
    heat = 1.0 / (1.0 + math.exp(-scale * (step - thresh)))
    freq = max_freq - (max_freq - base_freq) * heat
    return max(base_freq, round(freq))
def _component_guard(r, comp, step, c, fn, *a):
    """Run an optimizer side-component (sam/meta/bilevel) LOUDLY.

    No silent suppression (owner directive): failures are counted in
    r.component_failures (lands in saved results), warned with the count, and
    re-raised when c["strict_components"] is set — the tuner sets it so a
    hyperparameter config that crashes a component scores as a CRASH instead
    of quietly degrading to its base optimizer and masquerading as a DNF
    (which is how SuperGrok2 ran for weeks with SAM 100% dead)."""
    try:
        fn(*a); return True
    except Exception as e:
        n = r.component_failures[comp] = r.component_failures.get(comp, 0) + 1
        warnings.warn(f"{r.name} {comp} failed at step {step} (#{n}): {e}")
        if c.get("strict_components", False): raise
        return False

def _eval_log(r, step, m, tx, ty, vax, vay, tex, tey, c, st, pb):
    # Fused single-sync evaluation. The naive 3× evaluate() did SIX GPU→CPU
    # .item() round-trips per eval AND built autograd graphs for ~9.3K samples
    # (evaluate() lacked no_grad) — at the tuner's every-step cadence across a
    # 28-worker MPS fleet those sync stalls dominated (GPU ~25-40% util).
    # Identical math: same ops/dtypes, all metrics computed on-device, ONE
    # device→host transfer.
    p_ = c["p"]
    if c.get("fast_val_eval") and step % 10 != 0:
        # Between full evals: VAL-ONLY check (465 samples vs ~9.3K for the
        # triple) feeding just the early-stopper — preserves every-step
        # grokking_step resolution at ~1/9 the eval cost. Records/callbacks
        # stay on the 10-step grid.
        with torch.no_grad():
            logits = m(vax)
            va_t = torch.stack([F.cross_entropy(logits, vay),
                                (logits[:, :p_].argmax(-1) == vay).float().mean()])
            vl_f, va_f = va_t.cpu().tolist()
        stop_acc = va_f if c.get("early_stop_on", "test") == "val" else None
        if stop_acc is None:
            return False, None, None  # fast path only valid with val stopping
        return st.step(stop_acc, step), None, None
    with torch.no_grad():
        outs = []
        for x_, y_ in ((tx, ty), (vax, vay), (tex, tey)):
            logits = m(x_)
            outs.append(torch.stack([
                F.cross_entropy(logits, y_),
                (logits[:, :p_].argmax(-1) == y_).float().mean()]))
        (tl, ta), (vl, va), (tel, tea) = torch.stack(outs).cpu().tolist()
    r.steps.append(step); r.train_losses.append(tl); r.train_accs.append(ta)
    r.val_losses.append(vl); r.val_accs.append(va)
    r.test_losses.append(tel); r.test_accs.append(tea)
    pb.set_postfix({"trn":f"{ta:.3f}","val":f"{va:.3f}","tst":f"{tea:.3f}","tl":f"{tl:.3f}"}, refresh=False)
    # Optional per-eval observer (e.g. the Optuna tuner's pruning hook). May
    # raise (optuna.TrialPruned) to abort the run; absent in the race itself.
    cb = c.get("_eval_callback")
    if cb is not None:
        cb(step, ta, va, tea)
    stop_acc = va if c.get("early_stop_on", "test") == "val" else tea
    return st.step(stop_acc, step), tl, tel

# Map a TrainResult.name (the display name passed to _tr, e.g. "AdamW") to the
# optimizer KEY gemm_impl_for_cell expects ("adamw"). Only adamw cells have an
# L3-REAL path, so the wiring-guard degrade check only needs the adamw entry; the
# rest map to their keys for completeness (they always run eager, never L3).
OPT_KEY_BY_NAME = {
    "AdamW": "adamw", "NeuralGrok": "neuralgrok", "GrokAdamW": "grokadamw",
    "SuperGrok": "supergrok11", "SuperGrok1.5": "supergrok15",
    "SuperGrok2": "supergrok2", "Grokfast": "grokfast", "Muon": "muon",
    "Lion": "lion", "LookSAM": "looksam", "Prodigy": "prodigy",
}

def _record_train_path(r):
    """Compose r.train_path from the per-run path signals (task 2 WIRING GUARD).

    Reads the module globals LAST_L3_ENGINE (set by _try_fused_train_step on a
    successful L3 step → carries the ACTUAL engine) and LAST_L1_FIRED (set by
    _try_fused_step). Composes the human label the directive lists: "L3-TC bf16" /
    "L3-scalar fp32" / "L1+eager" / "eager".

    LOUD-DEGRADE: for an L3-REAL-CAPABLE cell+precision (decoder/vit×adamw at bf16,
    any ×adamw at fp32) that EXPECTED L3 but did not fire it, mark the path DEGRADED
    and emit a one-time stderr banner — UNLESS the stale-ABI latch tripped (a
    legitimate rebuild-pending soft-degrade, labelled as such, not a failure). The
    sweep CONTINUES (a hard raise would forfeit the roofline deliverable); the smoke
    gate asserts engine separately where "TC must fire" is load-bearing.
    """
    if LAST_L3_ENGINE is not None:
        r.train_path = LAST_L3_ENGINE["path"]
        return
    base = "L1+eager" if LAST_L1_FIRED else "eager"
    # Whether THIS cell could even take an L3-REAL path (the 3 adamw cells on sm_90).
    # has_l3_real is the decisive gate — without it, gemm_impl_for_cell's "scalar"
    # default would flag every non-adamw cell (muon/prodigy/…), which CORRECTLY run
    # eager. opt_key maps the display name → optimizer key for has_l3_real.
    opt_key = OPT_KEY_BY_NAME.get(r.name)
    l3_capable = opt_key is not None and has_l3_real(r.model_type, opt_key)
    expected_engine = (gemm_impl_for_cell(r.model_type, opt_key, r.matmul_precision)
                       if l3_capable else None)
    # DEGRADED fires ONLY for a genuine silent gate regression: an L3-capable cell,
    # fusion requested, not AMP, a wired precision (expected_engine set), that
    # declined L3 anyway AND is NOT the legitimate stale-ABI soft-degrade. Anything
    # else (non-L3 cell, use_fused off — e.g. the FLOP pass, AMP, tf32/fp16) is an
    # honest eager run, not a degrade.
    degraded = (l3_capable and r.use_fused_requested and not r.use_amp
                and expected_engine is not None and not _FUSED_ABI_STALE)
    if l3_capable and r.use_fused_requested and not r.use_amp \
            and expected_engine is not None and _FUSED_ABI_STALE:
        r.train_path = f"{base}(ABI-stale, rebuild pending)"
    elif degraded:
        want = _l3_path_label(expected_engine, r.matmul_precision)
        r.train_path = f"{base}(DEGRADED: expected {want})"
        msg = (f"[WIRING-GUARD] {r.name}/{r.model_type} @ {r.matmul_precision}: "
               f"expected {want} but ran {base} — L3-REAL path did NOT fire "
               f"(not ABI-stale). Roofline/race row is degraded; investigate.")
        print(msg, file=sys.stderr, flush=True)
    else:
        r.train_path = base


def _fin(r, st, step, t0, m, tex, tey, p=97):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _record_train_path(r)  # task 2: stamp the ACTUAL executed path onto the result
    r.wall_time=time.time()-t0; r.total_steps=step
    r.grokking_step=st.grokking_step; r.grokking_wall=st.grokking_wall
    r.stopping_reason=st.stopping_reason; r.stopping_step=st.stopping_step
    r.best_metric_acc=st.best_metric_acc  # [A4-M2] best stopper-criterion acc
    r.final_train_acc = r.train_accs[-1] if r.train_accs else 0.
    r.final_val_acc = r.val_accs[-1] if r.val_accs else 0.
    r.final_val_loss = r.val_losses[-1] if r.val_losses else 0.
    m.eval()
    with torch.no_grad():
        r.final_test_loss, r.final_test_acc = evaluate(m, tex, tey, p)
    m.train()
    r.val_test_gap = r.final_val_acc - r.final_test_acc
    # [A4-M3] TEST-confirm the first threshold crossing. Under val-stopping the
    # stopper triggers on val, but the SuperGrok meta-nets TRAIN on val, so a
    # val-only "grok" can be circular — confirm it transferred to the held-out
    # TEST split at the recorded eval nearest grokking_step (test_acc within
    # 0.05 of the stop threshold). Under test-stopping this is trivially True
    # whenever grokked (the criterion already IS test). recorded test_accs live
    # on the 10-step grid (r.steps/r.test_accs); fast_val_eval steps aren't
    # recorded, so we snap to the nearest recorded eval.
    if r.grokking_step is not None and r.steps and r.test_accs:
        gi = min(range(len(r.steps)), key=lambda i: abs(r.steps[i] - r.grokking_step))
        r.grokking_step_test_confirmed = bool(r.test_accs[gi] >= st.threshold - 0.05)
    else:
        r.grokking_step_test_confirmed = False
    return r
def _load(c, device, init_state):
    _apply_matmul_precision(c)  # set GEMM precision policy for this run
    m = build_model(c, device, c.get("compile_model", False))
    _mp = _resolve_matmul_precision(c)
    if _mp in ("fp8", "fp8e5m2", "int8"):
        from grokking_optimizers.lowprec import swap_linears_lowprec
        # swap AFTER build; weights load below targets the same (shared)
        # Parameter objects, so init/state keys are unaffected
        m._lowprec_report = swap_linears_lowprec(m, _mp)
    try: m.load_state_dict(copy.deepcopy(init_state), strict=True)
    except RuntimeError:
        raw = m._orig_mod if hasattr(m, "_orig_mod") else m
        raw.load_state_dict(copy.deepcopy(init_state), strict=True)
    return m
def _tr(name, c):
    # [A4-M4] capture the per-run resolved matmul precision + AMP flag so
    # save_json can record what precision actually ran (config c is not in
    # save_json's scope; the TrainResult carries it).
    # WIRING GUARD (task 2): reset the per-run path signals at run start so _fin
    # composes train_path from THIS run only (globals resolved at call-time, so the
    # forward reference to LAST_L3_ENGINE/LAST_L1_FIRED defined below is fine).
    global LAST_L3_ENGINE, LAST_L1_FIRED
    LAST_L3_ENGINE = None; LAST_L1_FIRED = False
    r = TrainResult(name, c["seed"], c.get("model_type","decoder"), c.get("frac_train",0.5),
                    c.get("val_ratio",0.10), matmul_precision=_resolve_matmul_precision(c),
                    use_amp=bool(c.get("use_amp", False)))
    r.use_fused_requested = bool(c.get("use_fused", True))  # task 2: guard input
    return r

# ── C++/CUDA fused optimizers (grokking_optimizers package) ────────────
from grokking_optimizers import (
    SuperGrok15, SuperGrok2, SuperGrok11,
    GrokAdamW, NeuralGrok, Prodigy, Grokfast, Lion, LookSAM, Muon,
)
from grokking_optimizers.dispatch import (
    detect_arch, has_fused, dispatch_fused, fused_optimizer_step,
    announce_fused_readiness, has_l3_real, fused_train_step,
    gemm_impl_for_cell, canonicalize_model,
)

# ── WIRING GUARD (owner directive task 2): the engine that ACTUALLY executed the
# last L3-REAL fused train step, set by _try_fused_train_step on every successful
# L3 step. None when the last step took the eager/L1 path. The roofline harness
# reads this (via the dispatch wrapper) to label the row with the REAL path
# (L3-TC bf16 / L3-scalar fp32) and pick the matching ceiling — never inferred from
# the requested precision. Format: dict(engine="wgmma"|"scalar", model=<canon>,
# precision=<str>, path=<human label>) or None. Module-global is safe: each train_*
# runs in one process, one model at a time.
LAST_L3_ENGINE = None
# Companion flag (task 2): True once the L1 fused optimizer-tail fired this run
# (whitelisted cell: eager fwd+bwd + fused tail). Lets _fin compose the path label
# without hand-editing every train_* loop. Reset at run start by _tr; the stale-ABI
# soft-degrade reason is tracked separately so the guard does not fire spuriously.
LAST_L1_FIRED = False


def _l3_path_label(engine, precision):
    """Human path label for TrainResult/run-JSON/roofline rows (task 2)."""
    if engine == "wgmma":
        return f"L3-TC {precision}"
    if engine == "scalar":
        return f"L3-scalar {precision}"
    return "L1+eager"

def _maybe_wrap_cuda_graph(opt, c):
    """No-op shim. CUDA Graph wrapping was removed in the post-refactor
    cleanup; the race is single-node and does not need graph capture."""
    return opt

# Process-wide stale-ABI latch for the fused optimizer path: set on the first
# pybind TypeError (extension predates the widened fused_step signature) so
# every subsequent step takes the eager path without re-attempting.
_FUSED_ABI_STALE = False

def _try_fused_step(model_name, opt_name, model, optimizer, x_batch, y_batch, c):
    """Run the L1 fused optimizer-tail megakernel for a whitelisted cell.

    CONTRACT CHANGE (lever (a)): this is now a post-backward OPTIMIZER STEP, not a
    fwd+bwd+opt launch. The caller MUST have already run the real forward +
    ``loss.backward()`` so every parameter carries its real ``p.grad``; this
    applies the canonical optimizer update in-place via ``ops.fused_step``
    (``opt_only=True`` → the L1 real-grad tail). Returns True if the fused step
    ran (the caller must then SKIP its own ``optimizer.step()`` / ``scaler.step``);
    False if the cell is not on the readiness whitelist (caller runs eager).

    Why not L3 (fwd+bwd+opt in one launch): the L3 megakernel runs an element-
    local SURROGATE model over the flat param blob (csrc/.../model_stages.cuh),
    NOT the real Transformer/ViT/Mamba graph, so its loss cannot match eager. The
    L1 tail consumes the real gradient and IS the eager optimizer step. See
    BUILD_AND_VALIDATE.md.

    State ownership: ``optimizer.step()`` is replaced, so the torch optimizer's
    own ``.state`` never fills. We keep a persistent per-parameter ``[m|v|extra]``
    buffer in a cache attached to the optimizer instance (allocated once per
    param) plus a step counter — reallocating per step would reset momentum and
    the run would never grok.
    """
    global _FUSED_ABI_STALE
    if _FUSED_ABI_STALE or not c.get("use_fused", True):
        return False
    try:
        # has_fused() is the readiness gate: True ONLY for whitelisted cells on a
        # GPU arch with a compiled fused TU. Keep it inside the guard so the fused
        # path always degrades to eager rather than crashing the run.
        if not has_fused(model_name, opt_name):
            return False
        announce_fused_readiness()  # one-time loud run-start banner (idempotent)
        # Persistent per-parameter state + step counter live on the optimizer
        # instance so they survive across iterations (the cache key is id(param)).
        cache = getattr(optimizer, "_fused_state_cache", None)
        if cache is None:
            cache = {}
            optimizer._fused_state_cache = cache
        step = getattr(optimizer, "_fused_step_counter", 0) + 1
        optimizer._fused_step_counter = step
        fused_optimizer_step(model_name, opt_name, model, optimizer,
                             state_cache=cache, step=step)
        global LAST_L1_FIRED
        LAST_L1_FIRED = True  # task 2: L1 fused tail fired this run (for _fin's path)
        return True
    except TypeError as e:
        # STALE-ABI GUARD: a pybind TypeError here means the compiled _ops
        # extension predates the widened fused_step signature (rebuild
        # pending). Without this catch the error escaped and CRASHED runs —
        # poisoning tuner trials with crash scores. Degrade LOUDLY ONCE per
        # process to the validated eager path; never retry (per-step retries
        # would spam and re-fail identically).
        if not _FUSED_ABI_STALE:
            _FUSED_ABI_STALE = True
            warnings.warn(
                "fused_step ABI mismatch (stale _ops build predates the widened "
                f"signature) — eager optimizer steps until rebuild: {e}")
        return False
    except (KeyError, NotImplementedError, ValueError, RuntimeError):
        # Any assembly/ABI problem (unbuilt extension, unsupported dtype/layout,
        # non-whitelisted cell) degrades to the eager path — never crash the run.
        return False

def _try_fused_train_step(model_name, opt_name, model, optimizer, x_batch,
                          y_batch, c):
    """PHASE 1+2 — run the TRUE L3 fused TRAIN step (real fwd+bwd+opt in ONE
    persistent megakernel) for an L3-REAL cell. Model-GENERIC: fires for whichever
    of (transformer_decoder|vit|mamba × adamw) has a compiled real fwd+bwd+opt
    kernel on this arch (has_l3_real gates it; PHASE 1 decoder, PHASE 2 vit+mamba).
    x_batch is the per-model input the cell expects — int token ids [B,seq] for
    decoder/mamba, float patches [B,16,49] for vit — and flows unchanged into
    fused_train_step, which packs it per model (see dispatch.py).

    Unlike ``_try_fused_step`` (the L1 post-backward optimizer tail, which needs
    the caller to have already run fwd+bwd), this REPLACES the eager forward +
    backward + optimizer.step() entirely: ONE persistent kernel runs the real
    model forward+backward AND AdamW — real model math, real optimizer math,
    ZERO intermediate launches (the owner rejected CUDA graphs; this is the path).

    Returns the training LOSS (a float, mean cross-entropy) if the fused train step
    ran — the caller then SKIPS its own fwd/bwd/step and logs THIS loss — or
    ``None`` if the L3-REAL path is NOT APPLICABLE for this cell/precision (caller
    falls back to eager + the L1 fused optimizer step). A None return is the honest
    eager path, NOT a degrade.

    WIRING GUARD (task 2): once the path-matched gate has SELECTED an engine for an
    L3-REAL cell, a launch failure RAISES (loud) rather than returning None — a
    silent degrade to eager here would let a roofline/race run secretly measure the
    wrong path. The ONLY soft-degrade is a stale-ABI TypeError (compiled _ops
    predates the widened fused_step signature → rebuild pending), which warns once
    and falls back process-wide. On every success this sets the module-global
    LAST_L3_ENGINE to the engine that ACTUALLY ran (dispatch.cpp has no silent
    fallback: a wgmma request runs wgmma or throws), so the path report is the real
    executed path, not the requested one.
    """
    global _FUSED_ABI_STALE, LAST_L3_ENGINE
    LAST_L3_ENGINE = None  # default: this step did NOT take L3 (set on success below)
    if _FUSED_ABI_STALE or not c.get("use_fused", True):
        return None
    # AVAILABILITY (legit-eager, not a degrade): not an L3-REAL cell on this arch.
    if not has_l3_real(model_name, opt_name):
        return None
    # PATH-MATCHED PRECISION GATE (owner directive task 1 — replaces the fp32-only
    # gate). The engine map picks the ACTUAL in-kernel GEMM engine for the run's
    # resolved precision: fp32 → scalar (all cells); bf16 → wgmma (decoder/vit TC
    # cells) / scalar (mamba carve-out). Fairness is preserved because the engine
    # MATCHES the run precision — a bf16 race runs the bf16 TC kernel (not fp32
    # in-kernel while competitors run bf16-eager, the old confound). fp16-AMP and
    # tf32 have no in-kernel carrier here → engine None → decline to eager (honest).
    if c.get("use_amp", False):
        return None
    precision = _resolve_matmul_precision(c)
    gemm_impl = gemm_impl_for_cell(model_name, opt_name, precision)
    if gemm_impl is None:
        return None  # unwired precision (fp16/tf32) — eager is the honest path
    # COMMITTED to the L3 path with a selected engine. From here a failure is LOUD.
    announce_fused_readiness()  # one-time loud run-start banner (idempotent)
    # Persistent flat-param + [m|v|extra]+loss state on the optimizer instance
    # (keyed by canonical model name), surviving across iterations — the megakernel
    # owns optimizer.step(), so the torch optimizer's .state never fills;
    # reallocating per step would reset momentum and never grok.
    cache = getattr(optimizer, "_fused_train_cache", None)
    if cache is None:
        cache = {}
        optimizer._fused_train_cache = cache
    step = getattr(optimizer, "_fused_train_counter", 0) + 1
    optimizer._fused_train_counter = step
    try:
        loss = fused_train_step(model_name, opt_name, model, optimizer,
                                x_batch, y_batch, state_cache=cache, step=step,
                                gemm_impl=gemm_impl)
    except TypeError as e:
        # STALE-ABI guard ONLY (compiled _ops predates the widened/gemm_impl
        # signature). This is the one legitimate soft-degrade: rebuild pending.
        # Degrade LOUDLY ONCE to eager process-wide; never retry.
        if not _FUSED_ABI_STALE:
            _FUSED_ABI_STALE = True
            warnings.warn(
                "fused_step ABI mismatch (stale _ops build predates the widened "
                f"signature) — eager train steps until rebuild: {e}")
        return None
    # Success → record the ACTUAL executed engine for the wiring guard / roofline.
    LAST_L3_ENGINE = dict(
        engine=gemm_impl, model=canonicalize_model(model_name),
        precision=precision, path=_l3_path_label(gemm_impl, precision))
    return loss

# ── 1. AdamW ──────────────────────────────────────────────────────────
def train_adamw(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("AdamW",c); m=_load(c,dev,init)
    opt=torch.optim.AdamW(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]), weight_decay=c["weight_decay"], fused=True)
    # AdamW baseline does not support use_grad_hooks (no _single_param_step API).
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    mtype=c.get("model_type","decoder")
    for step in (pb:=_pbar("AdamW",c["max_steps"],bp)):
        # PHASE 1 — TRUE L3 fused path for (decoder × adamw): ONE persistent
        # megakernel runs the REAL fwd+bwd+AdamW. If it ran, it returns the loss
        # and we SKIP the eager fwd/bwd/step entirely (the kernel already updated
        # the params in place). eval below stays eager/unchanged. Falls back to the
        # eager fwd+bwd + L1 fused tail when the L3-REAL kernel is unavailable
        # (non-decoder, AMP on, unbuilt TU, non-sm_90).
        l3_loss=_try_fused_train_step(mtype, "adamw", m, opt, tx, ty, c)
        if l3_loss is not None:
            loss=torch.as_tensor(l3_loss)  # for _eval_log's .item()-style logging
        else:
            # Real forward + backward (the megakernel L1 path is an OPTIMIZER step
            # on the real grad — NOT a fused fwd/bwd; see _try_fused_step).
            with _autocast(c):
                loss=F.cross_entropy(m(tx),ty)
            opt.zero_grad(); scaler.scale(loss).backward()
            # Whitelisted cell (decoder/vit/mamba × adamw): the L1 fused tail
            # applies the AdamW update in-place. On the fused path skip
            # scaler.step(opt) (the update already happened); still call
            # scaler.update() to advance the AMP scale.
            if _try_fused_step(mtype, "adamw", m, opt, tx, ty, c):
                scaler.update()
            else:
                scaler.step(opt); scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 2. NeuralGrok ─────────────────────────────────────────────────────
# NOTE: The NeuralGrok amplifier MLP is trained here via opt.train_amplifier_
# step on the held-out (val) split. The fused C++ kernel cannot call back into
# Python autograd, so it consumes a DETACHED snapshot of the amplifier weights;
# train_amplifier_step rebuilds the amplified update differentiably in Python
# (one-step lookahead val objective, mirroring SG11.meta_step), trains the
# amplifier, then marks the snapshot dirty so the next opt.step() re-extracts
# the freshly-trained weights. (AUDIT A3: before this, the amplifier's outer
# loss contained only the model — never the amplifier params — so the amplifier
# was frozen at random init and the snapshot refresh recopied unchanged
# weights. See NeuralGrok.train_amplifier_step for the full A3 writeup.)
def train_neuralgrok(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("NeuralGrok",c); m=_load(c,dev,init)
    opt=NeuralGrok(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha=c.get("neural_alpha",10.0),
        beta=c.get("neural_beta",4.0), num_layers=c.get("neural_layers",2),  # [A4-M6] align with OPTIMIZER_CONFIGS (kernel evaluates a 2-layer MLP)
        hidden_dim=c.get("neural_hidden",128), inner_steps=c.get("inner_steps",1),
        grad_clip=c.get("neural_grad_clip",1.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt.amplifier=opt.amplifier.to(dev)
    # External amplifier optimizer, passed explicitly to train_amplifier_step.
    # (NeuralGrok also owns an internal Adam as the AdamW-parity fallback; we
    # keep this explicit aopt so the race's training LR for the amplifier stays
    # visible at the call site.)
    aopt=opt.get_amplifier_optimizer(lr=1e-3)
    crit_ng=nn.CrossEntropyLoss()
    # Batch standardization (owner: identical batch per optimizer within a
    # model). The old 90/10 carve-out trained the MODEL on only 90% of the
    # train split (3772 of 4191) — the lone optimizer not at full batch. Model
    # gradients now use the FULL train batch like every other optimizer; the
    # amplifier's outer objective trains on the VAL split — the same held-out
    # convention the SuperGrok bilevel/meta nets already use. Mechanism intact,
    # batch equalized.
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("NeuralGrok",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        # AUDIT A3 FIX: the old block here — aopt.zero_grad(); aloss=neural_beta*
        # CE(m(vax),vay); scaler.scale(aloss).backward(); scaler.step(aopt) —
        # was a PERMANENT NO-OP: aloss's graph contained only the model m, and
        # the amplifier params were in neither that graph nor aopt-reachable
        # autograd, so the amplifier stayed frozen at random init and the
        # standalone mark_amplifier_dirty() refreshed unchanged weights forever.
        # train_amplifier_step rebuilds the amplified update DIFFERENTIABLY
        # through the amplifier (lookahead val objective, mirroring SG11.
        # meta_step) so the amplifier actually trains; it snapshots/restores
        # p.grad and calls mark_amplifier_dirty() itself. Grads are unscaled
        # above (like train_supergrok) so the method reads true-scale grads.
        # NOTE (AMP): this race defaults use_amp=False, so unscaled grads are
        # finite here. Under AMP a non-finite grad would propagate into the
        # amplifier objective; _component_guard catches exceptions but not a
        # silent NaN. If AMP is ever enabled for this optimizer, add the same
        # post-unscale isfinite skip train_supergrok uses before this call.
        _component_guard(r, "amplifier_step", step, c, opt.train_amplifier_step,
                         m, vax, vay, crit_ng, aopt)
        scaler.step(opt); scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 3. GrokAdamW ──────────────────────────────────────────────────────
def train_grokadamw(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("GrokAdamW",c); m=_load(c,dev,init)
    opt=GrokAdamW(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha=c.get("grokadamw_alpha",0.98),
        # PUBLISHED GrokAdamW: gamma (layer-wise β1 decay) and kappa
        # (grokking_signal_decay_rate, the α schedule rate) are WIRED mechanisms
        # (ref cognitivecomputations/grokadamw), not dead args. train/val loss
        # are fed below so α adapts from the grokking signal.
        gamma=c.get("grokadamw_gamma",0.1), kappa=c.get("grokadamw_kappa",0.1),
        lamb=c.get("grokadamw_lamb",2.0), grad_clip=c.get("grokadamw_grad_clip",1.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    alpha_freq=c.get("grokadamw_alpha_update_freq",50)
    for step in (pb:=_pbar("GrokAdamW",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward()
        # Feed the grokking signal: train_loss every alpha_freq/eval step, val CE
        # on the alpha cadence (mirrors the SuperGrok needs_metrics pattern). α_t
        # = alpha_init*exp(-kappa*signal) only updates when BOTH are present.
        kw={}
        needs_metrics=(step%alpha_freq==0) or (step%eval_every==0) or step==1
        if needs_metrics:
            with torch.no_grad():
                kw["train_loss"]=loss.item()
            if step%alpha_freq==0:
                with torch.no_grad():
                    kw["val_loss"]=F.cross_entropy(m(vax),vay).item()
        # scaler.step doesn't forward kwargs, so unscale + step directly (same as
        # the SuperGrok loops). Fall back to a plain step on a stale optimizer.
        scaler.unscale_(opt)
        try: opt.step(**kw)
        except TypeError: opt.step()
        scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 4. SuperGrok v1.1 ─────────────────────────────────────────────────
def train_supergrok(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    muf=c.get("supergrok_meta_update_freq",5)
    r=_tr("SuperGrok",c); m=_load(c,dev,init)
    opt=SuperGrok11(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha_init=c.get("supergrok_alpha",0.98),
        lamb=c.get("supergrok_lamb",1.0), gamma=c.get("supergrok_gamma",0.1),  # [A4-M6] align with OPTIMIZER_CONFIGS (lamb=1 identity default)
        kappa=c.get("supergrok_kappa",0.1), warmup_steps=c.get("supergrok_warmup",100),
        warmup_ramp=c.get("supergrok_warmup_ramp",100),
        gradient_clipping=c.get("supergrok_grad_clip",1.0),
        alpha_update_freq=c.get("supergrok_alpha_update_freq",50),
        gate_temperature=c.get("supergrok_gate_temp",5.0),
        zero_loss_threshold=c.get("supergrok_zero_loss_thresh",1e-4),
        zero_acc_threshold=c.get("supergrok_zero_acc_thresh",0.995),
        meta_hidden_dim=c.get("supergrok_meta_dim",32),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt.meta_net=opt.meta_net.to(dev)
    mopt=torch.optim.Adam(opt.meta_net.parameters(), lr=c.get("supergrok_meta_lr",1e-4))
    crit_sg=nn.CrossEntropyLoss()
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("SuperGrok",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            logits=m(tx); loss=F.cross_entropy(logits,ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        if c.get("use_amp", False):
            _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
            if _has_inf:
                scaler.update(); continue
        if step%muf==0:
            _component_guard(r, "meta_step", step, c, opt.meta_step, m, vax, vay, crit_sg, mopt, tx, ty)
        sam_freq = max(1, muf * 2)
        if hasattr(opt, 'sam_step') and step % sam_freq == 0 and opt._get_effective_sam_freq() < 999999:
            _component_guard(r, "sam_step", step, c, opt.sam_step, m, tx, ty, crit_sg)
        alpha_freq=c.get("supergrok_alpha_update_freq",50)
        kw={}
        needs_metrics=(step%alpha_freq==0) or (step%eval_every==0) or step==1
        if needs_metrics:
            with torch.no_grad():
                train_loss_val=loss.item()
                train_acc=(logits.detach()[:,:c["p"]].argmax(-1)==ty).float().mean().item()
            kw["train_loss"]=train_loss_val; kw["train_acc"]=train_acc
            if step%alpha_freq==0:
                with torch.no_grad():
                    vl_sg=F.cross_entropy(m(vax),vay).item()
                kw["val_loss"]=vl_sg
        try: opt.step(**kw)
        except TypeError: opt.step()
        scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 4b. SuperGrok v1.5 ────────────────────────────────────────────────
def train_supergrok15(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("SuperGrok1.5",c); m=_load(c,dev,init)
    opt=SuperGrok15(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha_init=c.get("supergrok15_alpha",0.98),
        lamb=c.get("supergrok15_lamb",2.0), gamma=c.get("supergrok15_gamma",0.1),
        kappa=c.get("supergrok15_kappa",0.1), warmup_steps=c.get("supergrok15_warmup",100),
        warmup_ramp=c.get("supergrok15_warmup_ramp",100),
        gradient_clipping=c.get("supergrok15_grad_clip",1.0),
        alpha_update_freq=c.get("supergrok15_alpha_update_freq",50),
        zero_loss_threshold=c.get("supergrok15_zero_loss_thresh",1e-4),
        zero_acc_threshold=c.get("supergrok15_zero_acc_thresh",0.995),
        meta_hidden_dim=c.get("supergrok15_meta_dim",32),
        sam_rho=c.get("supergrok15_sam_rho",0.05),
        gate_scale=c.get("supergrok15_gate_scale",20.0),
        gate_thresh=c.get("supergrok15_gate_thresh",0.8),
        sam_freq_min=c.get("supergrok15_sam_freq_min",3),
        sam_freq_max=c.get("supergrok15_sam_freq_max",20),
        sam_scale=c.get("supergrok15_sam_scale",20.0),
        sam_thresh=c.get("supergrok15_sam_thresh",0.85),
        bilevel_freq_min=c.get("supergrok15_bilevel_freq_min",5),
        bilevel_freq_max=c.get("supergrok15_bilevel_freq_max",30),
        bilevel_scale=c.get("supergrok15_bilevel_scale",20.0),
        bilevel_thresh=c.get("supergrok15_bilevel_thresh",0.9),
        wd_ramp=c.get("supergrok15_wd_ramp",4.0),
        wd_scale=c.get("supergrok15_wd_scale",20.0),
        wd_thresh=c.get("supergrok15_wd_thresh",0.9),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt.meta_net=opt.meta_net.to(dev)
    mopt=torch.optim.Adam(opt.meta_net.parameters(), lr=c.get("supergrok15_meta_lr",1e-4))
    crit_s15=nn.CrossEntropyLoss()
    alpha_freq=c.get("supergrok15_alpha_update_freq",50)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("SuperGrok1.5",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            logits=m(tx); loss=F.cross_entropy(logits,ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        if c.get("use_amp", False):
            _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
            if _has_inf:
                scaler.update(); continue
        sam_freq_eff=opt._get_effective_sam_freq()
        if sam_freq_eff < 999999 and step%sam_freq_eff==0:
            _component_guard(r, "sam_step", step, c, opt.sam_step, m, tx, ty, crit_s15)
        bilevel_freq_eff=opt._get_effective_bilevel_freq()
        if step%bilevel_freq_eff==0:
            _component_guard(r, "bilevel_step", step, c, opt.bilevel_step, m, tx, ty, vax, vay, crit_s15, mopt)
        kw={}
        needs_metrics=(step%alpha_freq==0) or (step%eval_every==0) or step==1
        if needs_metrics:
            with torch.no_grad():
                train_loss_val=loss.item()
                train_acc=(logits.detach()[:,:c["p"]].argmax(-1)==ty).float().mean().item()
            kw["train_loss"]=train_loss_val; kw["train_acc"]=train_acc
            if step%alpha_freq==0:
                with torch.no_grad():
                    vl_s15=F.cross_entropy(m(vax),vay).item()
                kw["val_loss"]=vl_s15
        try: opt.step(**kw)
        except TypeError: opt.step()
        scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 4c. SuperGrok v2 (Mamba-3 + PEER) ────────────────────────────────
def train_supergrok2(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("SuperGrok2",c); m=_load(c,dev,init)
    opt=SuperGrok2(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha_init=c.get("sg2_alpha",0.98),
        lamb=c.get("sg2_lamb",2.0), gamma=c.get("sg2_gamma",0.1),
        kappa=c.get("sg2_kappa",0.1), warmup_steps=c.get("sg2_warmup",100),
        warmup_ramp=c.get("sg2_warmup_ramp",100),
        gradient_clipping=c.get("sg2_grad_clip",1.0),
        d_model=c.get("sg2_d_model",8),
        d_state=c.get("sg2_d_state",16),
        mamba_expand=c.get("sg2_mamba_expand",2),
        num_peer_heads=c.get("sg2_num_peer_heads",4),
        num_experts=c.get("sg2_num_experts",144),
        expert_hidden=c.get("sg2_expert_hidden",16),
        gru_hidden=c.get("sg2_gru_hidden",4),
        meta_rescale=c.get("sg2_meta_rescale",0.1),
        recycle_interval=c.get("sg2_recycle_interval",100),
        recycle_threshold=c.get("sg2_recycle_threshold",0.001),
        alpha_update_freq=c.get("sg2_alpha_update_freq",50),
        zero_loss_threshold=c.get("sg2_zero_loss_thresh",1e-4),
        zero_acc_threshold=c.get("sg2_zero_acc_thresh",0.995),
        sam_rho=c.get("sg2_sam_rho",0.05),
        gate_scale=c.get("sg2_gate_scale",20.0), gate_thresh=c.get("sg2_gate_thresh",0.8),
        sam_freq_min=c.get("sg2_sam_freq_min",3), sam_freq_max=c.get("sg2_sam_freq_max",20),
        sam_scale=c.get("sg2_sam_scale",20.0), sam_thresh=c.get("sg2_sam_thresh",0.85),
        bilevel_freq_min=c.get("sg2_bilevel_freq_min",5), bilevel_freq_max=c.get("sg2_bilevel_freq_max",30),
        bilevel_scale=c.get("sg2_bilevel_scale",20.0), bilevel_thresh=c.get("sg2_bilevel_thresh",0.9),
        wd_ramp=c.get("sg2_wd_ramp",4.0), wd_scale=c.get("sg2_wd_scale",20.0),
        wd_thresh=c.get("sg2_wd_thresh",0.9),
        sam_enable_threshold=c.get("sg2_sam_enable_threshold",0.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt.meta_net=opt.meta_net.to(dev)
    mopt=torch.optim.Adam(opt.meta_net.parameters(), lr=c.get("sg2_meta_lr",1e-4))
    crit_s2=nn.CrossEntropyLoss()
    alpha_freq=c.get("sg2_alpha_update_freq",50)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("SuperGrok2",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            logits=m(tx); loss=F.cross_entropy(logits,ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        if c.get("use_amp", False):
            _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
            if _has_inf:
                scaler.update(); continue
        sam_freq_eff=opt._get_effective_sam_freq()
        if sam_freq_eff < 999999 and step%sam_freq_eff==0:
            _component_guard(r, "sam_step", step, c, opt.sam_step, m, tx, ty, crit_s2)
        bilevel_freq_eff=opt._get_effective_bilevel_freq()
        if step%bilevel_freq_eff==0:
            _component_guard(r, "bilevel_step", step, c, opt.bilevel_step, m, tx, ty, vax, vay, crit_s2, mopt)
        kw={}
        needs_metrics=(step%alpha_freq==0) or (step%eval_every==0) or step==1
        if needs_metrics:
            with torch.no_grad():
                train_loss_val=loss.item()
                train_acc=(logits.detach()[:,:c["p"]].argmax(-1)==ty).float().mean().item()
            kw["train_loss"]=train_loss_val; kw["train_acc"]=train_acc
            if step%alpha_freq==0:
                with torch.no_grad():
                    vl_s2=F.cross_entropy(m(vax),vay).item()
                kw["val_loss"]=vl_s2
        try: opt.step(**kw)
        except TypeError: opt.step()
        scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 5. Grokfast ───────────────────────────────────────────────────────
def train_grokfast(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("Grokfast",c); m=_load(c,dev,init)
    opt=Grokfast(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], grokfast_alpha=c.get("grokfast_alpha",0.98),
        grokfast_lamb=c.get("grokfast_lamb",2.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("Grokfast",c["max_steps"],bp)):
        # NO L3-TC path: grokfast is BLOCKED from the bf16 TC driver — the optimizer
        # cold-starts ema=grad0 (grokfast.py) while the TC P3 state cache inits ema=0,
        # so the in-kernel ema slice diverges from the real optimizer for ~1/(1-alpha)
        # steps (state-gate: ema rel 0.98 at step 1). Converting needs a step==1 ema
        # init in the shared TC P3 (cold-start-STAGED). Until then: real eager step.
        # [A4-M5] L1 fused tail integration requires the adamw/lion post-backward
        # structure — do not re-add the pre-forward continue pattern.
        with _autocast(c):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 6. Muon ───────────────────────────────────────────────────────────
def train_muon(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("Muon",c); m=_load(c,dev,init)
    muon_params, adam_params = [], []
    for n,p in m.named_parameters():
        if p.requires_grad: (muon_params if p.ndim==2 else adam_params).append(p)
    opt=Muon(muon_params, params_1d=adam_params if adam_params else None,
        lr=c.get("muon_lr",0.02), momentum=c.get("muon_momentum",0.95),
        weight_decay=c["weight_decay"], adamw_lr=c["lr"],
        adamw_betas=(c["beta1"],c["beta2"]),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("Muon",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 7. Lion ───────────────────────────────────────────────────────────
def train_lion(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("Lion",c); m=_load(c,dev,init)
    opt=Lion(m.parameters(), lr=c.get("lion_lr",3e-4),
        betas=(c["beta1"],0.99), weight_decay=c.get("lion_wd",3.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    mtype=c.get("model_type","decoder")
    for step in (pb:=_pbar("Lion",c["max_steps"],bp)):
        # L3-TC path (owner baseline directive): for a wgmma-wired (model × lion)
        # cell the bf16 tensor-core megakernel runs the REAL fwd+bwd AND the Lion
        # tail (apply_optimizer<Lion>) in ONE persistent kernel — identical fwd+bwd
        # to the adamw TC cell, only the per-element tail differs. If it ran it
        # returns the loss and we SKIP the eager fwd/bwd/step. Otherwise: eager
        # fwd+bwd + the whitelisted L1 fused tail (lion uses only the m=exp_avg
        # slice of [m|v|extra]).
        l3_loss=_try_fused_train_step(mtype, "lion", m, opt, tx, ty, c)
        if l3_loss is not None:
            loss=torch.as_tensor(l3_loss)
        else:
            with _autocast(c):
                loss=F.cross_entropy(m(tx),ty)
            opt.zero_grad(); scaler.scale(loss).backward()
            if _try_fused_step(mtype, "lion", m, opt, tx, ty, c):
                scaler.update()           # fused path: skip scaler.step(opt)
            else:
                scaler.step(opt); scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 9. LookSAM ───────────────────────────────────────────────────────
def train_looksam(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("LookSAM",c); m=_load(c,dev,init)
    k=c.get("looksam_k",5)
    opt=LookSAM(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], rho=c.get("looksam_rho",0.05),
        k=k, alpha=c.get("looksam_alpha",0.7),
        use_grad_hooks=c.get("use_grad_hooks",False))
    crit_ls=nn.CrossEntropyLoss()
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("LookSAM",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        if c.get("use_amp", False):
            _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
            if _has_inf:
                scaler.update(); continue
        if opt.should_sam_step():
            opt.sam_step(m, tx, ty, crit_ls)
        opt.step()
        scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── 10. Prodigy ───────────────────────────────────────────────────────
def train_prodigy(c, init, tx, ty, vax, vay, tex, tey, dev, bp=0):
    r=_tr("Prodigy",c); m=_load(c,dev,init)
    opt=Prodigy(m.parameters(), lr=c.get("prodigy_lr",1.0), weight_decay=c["weight_decay"],
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time(); eval_every=c.get("eval_every",100)
    for step in (pb:=_pbar("Prodigy",c["max_steps"],bp)):
        # [A4-M5] L1 fused tail integration for this optimizer requires the
        # adamw/lion post-backward structure (see train_adamw) — do not re-add
        # the pre-forward continue pattern (it skips side-steps and eval).
        with _autocast(c):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%eval_every==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vax,vay,tex,tey,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0,m,tex,tey,c["p"])

# ── Registry ──────────────────────────────────────────────────────────
OPTIMIZER_REGISTRY = {
    "adamw":train_adamw, "neuralgrok":train_neuralgrok,
    "grokadamw":train_grokadamw, "supergrok":train_supergrok,
    "supergrok15":train_supergrok15, "supergrok2":train_supergrok2,
    "grokfast":train_grokfast, "muon":train_muon,
    "lion":train_lion,
    "looksam":train_looksam, "prodigy":train_prodigy,
}
COLORS = {
    "adamw":"#1f77b4","neuralgrok":"#ff7f0e","grokadamw":"#2ca02c",
    "supergrok":"#d62728","supergrok15":"#ff4444","supergrok2":"#ff8800",
    "grokfast":"#9467bd","muon":"#8c564b",
    "lion":"#e377c2","looksam":"#bcbd22","prodigy":"#17becf",
}
DISPLAY_NAMES = {
    "adamw":"AdamW","neuralgrok":"NeuralGrok","grokadamw":"GrokAdamW",
    "supergrok":"SuperGrok","supergrok15":"SuperGrok1.5","supergrok2":"SuperGrok2",
    "grokfast":"Grokfast","muon":"Muon",
    "lion":"Lion","looksam":"LookSAM","prodigy":"Prodigy",
}
MODEL_COLORS = {"decoder":"#1f77b4","vit":"#ff7f0e","mamba":"#2ca02c"}

# ─────────────────────────────────────────────────────────────────────
#  PART 3: PLOTTING + SUMMARY + RUNNERS
# ─────────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def _ema_smooth(values, alpha=0.9):
    """Exponential moving average smoothing."""
    if not values: return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed

def _is_crashed(r):
    """[A4-H3] A stub crash result has no recorded curve (steps==[]) and a
    CRASHED: stopping_reason — distinguish it from a genuine DNF (which has a
    full curve but never grokked)."""
    return (not r.steps) and isinstance(r.stopping_reason, str) and r.stopping_reason.startswith("CRASHED:")

def _live(runs):
    """[A4-H3] Drop crash stubs (steps==[] / CRASHED reason) from a runs list so
    curve/bar aggregations operate only on runs that actually produced data;
    crashes are counted separately (save_json _meta.crashes, print_summary)."""
    return [r for r in runs if not _is_crashed(r)]

def _grok_rate(runs):
    """[A4-H3] Grok success rate over LIVE runs (crash stubs excluded from both
    numerator and denominator); np.nan when every seed crashed so the heatmap
    renders it as 0 via nan_to_num rather than dividing by zero."""
    lv=_live(runs)
    if not lv: return float("nan")
    return sum(1 for r in lv if r.grokking_step)/len(lv)

def _interpolate_runs(runs, attr, num_points=500):
    """Interpolate multiple runs onto a common step grid, return mean ± std."""
    runs = _live(runs)  # [A4-H3] skip crashed runs (empty curves)
    if not runs: return [], [], []
    steps_avail = [r.steps[-1] for r in runs if r.steps]
    if not steps_avail: return [], [], []
    max_step = max(steps_avail)
    if max_step == 0: return [], [], []
    common_steps = np.linspace(0, max_step, num_points)
    interpolated = []
    for r in runs:
        if not r.steps: continue
        vals = np.array(getattr(r, attr))
        steps = np.array(r.steps)
        interp = np.interp(common_steps, steps, vals)
        interpolated.append(interp)
    if not interpolated: return [], [], []
    arr = np.array(interpolated)
    return common_steps, arr.mean(axis=0), arr.std(axis=0)

def plot_comparison(rbo, save_dir="results", thresh=0.95, ft=0.5, model_type="decoder", suffix=""):
    os.makedirs(save_dir, exist_ok=True)
    mt_label=MODEL_LABELS.get(model_type,model_type); task_label=TASK_LABELS.get(model_type,"")
    ema_alpha = 0.92  # smoothing factor

    # ── Curve plots with mean ± std band ────────────────────────────
    fig, axes = plt.subplots(2,2,figsize=(18,13))
    fig.suptitle(f"Grokking Race — {task_label} | {mt_label}\n"
                 f"train/test={ft*100:.0f}/{(1-ft)*100:.0f} | {thresh*100:.0f}% threshold",
                 fontsize=14, fontweight="bold")
    for row,col,attr,title,logy in [(0,0,"train_accs","Train Acc",False),(0,1,"val_accs","Val Acc",False),
                                     (1,0,"train_losses","Train Loss",True),(1,1,"val_losses","Val Loss",True)]:
        ax=axes[row,col]
        for name, runs in rbo.items():
            clr=COLORS.get(name,"#888888")
            dname=DISPLAY_NAMES.get(name,name)
            steps, mean, std = _interpolate_runs(runs, attr)
            if len(steps) == 0: continue
            # EMA smooth the mean
            mean_smooth = np.array(_ema_smooth(list(mean), ema_alpha))
            ax.plot(steps, mean_smooth, label=dname, color=clr, linewidth=2)
            if len(runs) > 1:
                ax.fill_between(steps,
                    np.clip(mean_smooth - std, 1e-8 if logy else -0.05, None),
                    mean_smooth + std,
                    color=clr, alpha=0.15)
        ax.set_xlabel("Steps"); ax.set_ylabel(title); ax.set_title(title)
        if logy: ax.set_yscale("log")
        else: ax.set_ylim(-0.05,1.05)
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"curves{suffix}.png"), dpi=150, bbox_inches="tight"); plt.close()

    # ── Race bar chart ──────────────────────────────────────────────
    # [A4-H3] rank/aggregate over LIVE runs only (crash stubs have empty curves);
    # a name with only crashes sorts last (1e9) and renders a zero bar.
    def _gw_key(n):
        lv=_live(rbo[n]); return np.mean([r.grokking_wall or r.wall_time for r in lv]) if lv else 1e9
    ns = sorted(rbo.keys(), key=_gw_key)
    fig2,(ax1,ax2) = plt.subplots(1,2,figsize=(18,6))
    fig2.suptitle(f"Grokking Race — {mt_label}  [{task_label}] | split={ft*100:.0f}/{(1-ft)*100:.0f}", fontsize=13, fontweight="bold")
    for i,name in enumerate(ns):
        runs=_live(rbo[name])
        wt=[r.grokking_wall or r.wall_time for r in runs] or [0.0]
        gs=[r.grokking_step or r.total_steps for r in runs] or [0]
        clr=COLORS.get(name,"#888888"); dname=DISPLAY_NAMES.get(name,name)
        nogrok=any(r.grokking_wall is None for r in runs) or not runs
        ax1.barh(i, np.mean(wt), xerr=np.std(wt) if len(wt)>1 else 0, color=clr, edgecolor="black", alpha=0.85, capsize=4)
        ax1.text(np.mean(wt)+(np.std(wt) if len(wt)>1 else 0)+0.3, i, f"{np.mean(wt):.1f}s"+(" ✗" if nogrok else " ✓"), va="center", fontsize=9)
        ax2.barh(i, np.mean(gs), xerr=np.std(gs) if len(gs)>1 else 0, color=clr, edgecolor="black", alpha=0.85, capsize=4)
        ax2.text(np.mean(gs)+(np.std(gs) if len(gs)>1 else 0)+50, i, f"{np.mean(gs):,.0f}"+(" ✗" if nogrok else " ✓"), va="center", fontsize=9)
    for ax in [ax1,ax2]:
        ax.set_yticks(range(len(ns))); ax.set_yticklabels([DISPLAY_NAMES.get(n,n) for n in ns]); ax.invert_yaxis(); ax.grid(axis="x",alpha=0.3)
    ax1.set_xlabel("Wall-Clock (s)"); ax1.set_title("⏱ Time to Grok"); ax2.set_xlabel("Steps"); ax2.set_title("Steps to Grok")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"race{suffix}.png"), dpi=150, bbox_inches="tight"); plt.close()

def print_summary(rbo, total_wall=None, model_type="decoder", frac_train=0.5):
    w=105; mt_label=MODEL_LABELS.get(model_type,model_type)
    print("\n"+"="*w); print(f"  🏁  GROKKING RACE — {mt_label} | split={frac_train*100:.0f}/{(1-frac_train)*100:.0f}"); print("="*w)
    # [A4-H3] rank/aggregate over LIVE runs (crash stubs excluded); a name with
    # only crashes sorts last. Crash counts are reported loudly below the table.
    crash_counts={name: sum(1 for r in runs if _is_crashed(r)) for name,runs in rbo.items()}
    def _rank_key(kv):
        lv=_live(kv[1]); return np.mean([r.grokking_wall or 1e9 for r in lv]) if lv else 1e9
    ranked=sorted(rbo.items(), key=_rank_key)
    multi=any(len(_live(v))>1 for v in rbo.values())
    hdr=f"  {'#':>2} {'Optimizer':<14} {'Grok Wall (s)':>14} {'Grok Steps':>12} {'Total Steps':>12} {'Val Acc':>9} {'Status':>8}"
    if multi: hdr+=f" {'Seeds':>6}"
    print(hdr); print("  "+"-"*(w-2))
    medals=["🥇","🥈","🥉"]+["  "]*20
    for rank,(name,runs) in enumerate(ranked):
        dname=DISPLAY_NAMES.get(name,name)
        lv=_live(runs)
        if not lv:
            # All seeds crashed — show CRASHED row, skip the (undefined) stats.
            line=f"  {medals[rank]} {dname:<14} {'—':>14} {'—':>12} {'—':>12} {'—':>9} {'✗ CRASH':>8}"
            if multi: line+=f" {len(runs):>6}"
            print(line); continue
        gw=[r.grokking_wall or r.wall_time for r in lv]; va=[r.final_val_acc for r in lv]
        nogrok=any(r.grokking_wall is None for r in lv)
        line=(f"  {medals[rank]} {dname:<14} {np.mean(gw):>14.2f} "
              f"{np.mean([r.grokking_step or r.total_steps for r in lv]):>12,.0f} "
              f"{np.mean([r.total_steps for r in lv]):>12,.0f} "
              f"{np.mean(va):>9.4f} {'✗ DNF' if nogrok else '✓ GROK':>8}")
        if multi: line+=f" {len(lv):>6}"
        print(line)
    print("  "+"-"*(w-2))
    # [A4-H3] LOUD per-optimizer crash count — a crashed config must never hide.
    total_crashes=sum(crash_counts.values())
    if total_crashes:
        print(f"  ⚠ CRASHES ({total_crashes} run(s) crashed — counted, excluded from stats above):")
        for name in sorted(crash_counts):
            if crash_counts[name]:
                ex=next((r.stopping_reason for r in rbo[name] if _is_crashed(r)), "")
                print(f"      ✗ {DISPLAY_NAMES.get(name,name):<14} {crash_counts[name]} crash(es)   e.g. {ex}")
        print("  "+"-"*(w-2))
    if total_wall: print(f"  Pipeline wall: {total_wall:.1f}s"); print("="*w)

def _crash_stub(name, cfg, exc):
    """[A4-H3] Build a stub TrainResult for a run that CRASHED so it is counted,
    not dropped. Empty curves, total_steps=0, stopping_reason carries the
    exception. Every results-list consumer guards on steps==[] (see _is_crashed,
    print_summary, plot_comparison, _interpolate_runs, _plot_full_sweep)."""
    r = _tr(name, cfg)
    r.stopping_reason = f"CRASHED: {type(exc).__name__}: {exc}"[:300]
    r.total_steps = 0
    return r

def save_json(rbo, save_dir="results", total_wall=None, model_type="decoder", frac_train=0.5):
    os.makedirs(save_dir, exist_ok=True)
    # [A4-M4] record the default matmul precision policy; [A4-H3] per-optimizer
    # crash counts so a crashed config is COUNTED, never silently dropped.
    crashes={name: sum(1 for r in runs if _is_crashed(r)) for name,runs in rbo.items()}
    d={"_meta":{"total_wall":total_wall,"model_type":model_type,"frac_train":frac_train,
                "matmul_precision_default":DEFAULT_CONFIG.get("matmul_precision"),
                "crashes":crashes}}
    for name,runs in rbo.items():
        d[name]=[{"seed":r.seed,"steps":r.steps,"train_losses":r.train_losses,"train_accs":r.train_accs,
            "val_losses":r.val_losses,"val_accs":r.val_accs,
            "test_losses":r.test_losses,"test_accs":r.test_accs,
            "wall_time":r.wall_time,"total_steps":r.total_steps,
            "grokking_step":r.grokking_step,"grokking_wall":r.grokking_wall,
            "grokking_step_test_confirmed":r.grokking_step_test_confirmed,  # [A4-M3]
            "best_metric_acc":r.best_metric_acc,  # [A4-M2]
            "final_val_acc":r.final_val_acc,"final_train_acc":r.final_train_acc,
            "final_test_acc":r.final_test_acc,"final_test_loss":r.final_test_loss,
            "final_val_loss":r.final_val_loss,"stopping_reason":r.stopping_reason,
            "stopping_step":r.stopping_step,"val_test_gap":r.val_test_gap,
            "val_ratio":r.val_ratio,
            "component_failures":dict(getattr(r,"component_failures",{}) or {}),  # [A4-H2]
            "matmul_precision":r.matmul_precision,  # [A4-M4]
            "use_amp":bool(r.use_amp),  # [A4-M4]
            "train_path":getattr(r,"train_path",None),  # task 2: ACTUAL executed path
            "model_type":r.model_type,"frac_train":r.frac_train} for r in runs]
    with open(os.path.join(save_dir,f"results_{model_type}_ft{int(frac_train*100)}.json"),"w") as f:
        json.dump(d,f,indent=2)

# ── Multi-GPU worker ──────────────────────────────────────────────────
def _gpu_worker(gpu_id, task_queue, base, merged, result_queue, worker_id):
    """Pull tasks from a shared queue and run them on a specific GPU.

    Each worker is an independent process with exclusive GPU access.
    Data and init states are lazily created on the assigned GPU to avoid
    cross-device tensor issues.

    Args:
        gpu_id: CUDA device index (e.g. 0, 1, 2, 3)
        task_queue: mp.Queue of (optimizer_name, seed) tuples; None = stop
        base: base config dict
        merged: dict of {optimizer_name: merged_config}
        result_queue: mp.Queue for returning (name, seed, TrainResult or None)
        worker_id: integer ID for logging
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        print(f"  [GPU {gpu_id}] Worker {worker_id} started on {device}")

        # Lazily populated cache of seed data on this GPU
        seed_data = {}

        while True:
            task = task_queue.get()
            if task is None:  # poison pill
                break
            name, s = task

            if s not in seed_data:
                torch.manual_seed(s); np.random.seed(s)
                tx, ty, vax, vay, tex, tey = make_data_for_task(base, s)
                tx, ty = tx.to(device), ty.to(device)
                vax, vay = vax.to(device), vay.to(device)
                tex, tey = tex.to(device), tey.to(device)
                ctmp = dict(base); ctmp["seed"] = s
                ist = get_init_state(ctmp, device)
                seed_data[s] = (tx, ty, vax, vay, tex, tey, ist)

            cfg = dict(merged[name]); cfg["seed"] = s
            tx, ty, vax, vay, tex, tey, ist = seed_data[s]
            try:
                res = OPTIMIZER_REGISTRY[name](cfg, ist, tx, ty, vax, vay, tex, tey, device, 0)
                result_queue.put((name, s, res))
                grokked = res.grokking_step is not None
                status = f"✓ grokked step {res.grokking_step}" if grokked else f"✗ DNF"
                print(f"  [GPU {gpu_id}] {DISPLAY_NAMES.get(name,name)} seed={s}: {status} ({res.wall_time:.1f}s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [GPU {gpu_id}] ✗ {name} seed={s} FAILED: {e}")
                # [A4-H3] send a CRASH STUB (empty curves) instead of None so the
                # collector counts the crash rather than dropping the run.
                result_queue.put((name, s, _crash_stub(name, cfg, e)))

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [GPU {gpu_id}] Worker {worker_id} crashed: {e}")


# ── run_pipeline ──────────────────────────────────────────────────────
def run_pipeline(optimizers=None, optimizer_configs=None, seeds=None,
                 compile_model=False, parallel=True, max_steps=None,
                 lr=None, weight_decay=None, threshold=None,
                 frac_train=None, val_ratio=None, seed=None, device_str=None,
                 save_dir="results", model_type=None, gpu_ids=None,
                 use_amp=False, model_scale=None,
                 early_stop_max_steps=None, eval_every=None):
    base=dict(DEFAULT_CONFIG)
    # [A4-M8] log_every dropped: dead config key, never read by any train loop.
    for k,v in [("max_steps",max_steps),("lr",lr),("weight_decay",weight_decay),
                ("early_stop_threshold",threshold),
                ("frac_train",frac_train),("val_ratio",val_ratio),("seed",seed),
                ("model_type",model_type),("early_stop_max_steps",early_stop_max_steps),
                ("eval_every",eval_every)]:
        if v is not None: base[k]=v
    # Auto-override: 10/90 split uses val_ratio=0.05 if not explicitly set
    if val_ratio is None and base["frac_train"] == 0.10:
        base["val_ratio"] = 0.05
    base["compile_model"]=compile_model
    base["use_amp"]=use_amp
    if model_scale is not None and model_scale in MODEL_SCALES:
        base.update(MODEL_SCALES[model_scale])
    cl=base.get("chain_length",3); base["seq_len"]=2*cl+2

    # ── Device selection ──────────────────────────────────────────────
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = False
    if gpu_ids is not None and len(gpu_ids) > 1 and n_gpus >= 2:
        use_multi_gpu = True
        # Validate requested GPUs exist
        gpu_ids = [g for g in gpu_ids if g < n_gpus]
        if len(gpu_ids) < 2:
            use_multi_gpu = False
    elif gpu_ids is not None and len(gpu_ids) == 1:
        device_str = f"cuda:{gpu_ids[0]}"

    if not use_multi_gpu:
        device=(torch.device(device_str) if device_str
                else torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if hasattr(torch.backends,"mps") and torch.backends.mps.is_available()
                else torch.device("cpu"))
    else:
        device = torch.device(f"cuda:{gpu_ids[0]}")  # for printing / param count

    if optimizers is None: optimizers=list(OPTIMIZER_REGISTRY.keys())
    if optimizer_configs is None: optimizer_configs={}
    if seeds is None: seeds=[base["seed"]]
    os.makedirs(save_dir, exist_ok=True)
    mt=base.get("model_type","decoder"); mt_label=MODEL_LABELS.get(mt,mt)

    if use_multi_gpu:
        dev_str = f"{len(gpu_ids)} GPUs: {gpu_ids}"
    else:
        dev_str = str(device)
    vr=base["val_ratio"]; ft=base["frac_train"]
    train_pct=ft*(1-vr)*100; val_pct=ft*vr*100; test_pct=(1-ft)*100
    print(f"\n{'='*60}\n  Model : {mt_label}\n  Task  : {TASK_LABELS.get(mt,'')}\n"
          f"  Device: {dev_str}\n  Split : {train_pct:.1f}/{val_pct:.1f}/{test_pct:.1f} (train/val/test)\n"
          f"  Seeds : {seeds}\n  Max   : {base['max_steps']:,} steps | Early-stop: {base.get('early_stop_max_steps',base['max_steps']):,}\n{'='*60}")
    merged={n: _merge(base, optimizer_configs.get(n)) for n in optimizers}
    for n in merged: merged[n]["model_type"]=mt; merged[n]["frac_train"]=base["frac_train"]; merged[n]["val_ratio"]=base["val_ratio"]; merged[n]["seq_len"]=base["seq_len"]
    tasks=[(n,s) for n in optimizers for s in seeds]
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    tmp_dev = device if not use_multi_gpu else torch.device(f"cuda:{gpu_ids[0]}")
    torch.manual_seed(seeds[0]); np.random.seed(seeds[0])
    tx0,ty0,vax0,vay0,tex0,tey0 = make_data_for_task(base, seeds[0])
    print(f"Train: {tx0.shape[0]:,} | Val: {vax0.shape[0]:,} | Test: {tex0.shape[0]:,} | x shape: {list(tx0.shape)}")
    npar=sum(p.numel() for p in _raw_model(base,tmp_dev).parameters())
    print(f"Params ({mt}): {npar:,}")
    del tx0, ty0, vax0, vay0, tex0, tey0

    results_by_opt=defaultdict(list); total_t0=time.time()

    # ══════════════════════════════════════════════════════════════════
    #  MULTI-GPU PATH
    # ══════════════════════════════════════════════════════════════════
    if use_multi_gpu:
        n_workers = len(gpu_ids)
        print(f"\n  ▸ Multi-GPU mode: distributing {total_tasks} tasks across {n_workers} GPUs")

        # Shared work queue: all tasks go into one queue, workers pull dynamically.
        # This ensures maximum GPU utilization — fast GPUs pick up more tasks.
        task_queue = MPQueue()
        for task in tasks:
            task_queue.put(task)
        # Poison pills — one per worker so each knows when to stop
        for _ in range(n_workers):
            task_queue.put(None)

        print(f"    {total_tasks} tasks in shared queue across {n_workers} GPUs: {gpu_ids}")

        # Spawn workers
        result_queue = MPQueue()
        workers = []
        for i in range(n_workers):
            p = mp.Process(
                target=_gpu_worker,
                args=(gpu_ids[i], task_queue, base, merged, result_queue, i),
                daemon=False,
            )
            workers.append(p)
            p.start()

        # Collect results
        received = 0
        while received < total_tasks:
            try:
                name, s, res = result_queue.get(timeout=7200)  # 2hr max per task
                received += 1
                # [A4-H3] crash stubs (and any None from a legacy worker) are
                # COUNTED, never dropped: stubs go into results_by_opt so the
                # JSON summary's per-optimizer crash count sees them, but they
                # are logged as errors (not "completed") in the progress board.
                if res is not None and not _is_crashed(res):
                    results_by_opt[name].append(res)
                    grokked = res.grokking_step is not None
                    with _PROGRESS_LOCK:
                        _PROGRESS["completed"].append({
                            "name":name, "model":mt, "split":base["frac_train"],
                            "seed":s, "grokked":grokked, "wall_time":res.wall_time,
                            "val_acc":res.final_val_acc
                        })
                    if grokked:
                        _ntfy(f"✓ {name} grokked at step {res.grokking_step} ({res.grokking_wall:.1f}s) | {mt} ft={base['frac_train']} seed={s}",
                               title="Grokked!", tags="white_check_mark")
                else:
                    if res is not None:
                        results_by_opt[name].append(res)  # keep the crash stub for the count
                    reason = res.stopping_reason if res is not None else "worker returned None"
                    with _PROGRESS_LOCK:
                        _PROGRESS["errors"].append({"name":name, "seed":s, "error":reason})
                _update_progress(current_run=received, current_task=f"[multi-GPU] {received}/{total_tasks}")
            except Exception as e:
                print(f"  ✗ Queue error: {e}")
                break

        # Wait for all workers to finish
        for p in workers:
            p.join(timeout=60)
            if p.is_alive():
                print(f"  ⚠ Worker {p.pid} still alive, terminating")
                p.terminate()

    # ══════════════════════════════════════════════════════════════════
    #  SINGLE-GPU / CPU PATH (original sequential)
    # ══════════════════════════════════════════════════════════════════
    else:
        seed_data={}
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            tx,ty,vax,vay,tex,tey=make_data_for_task(base, s)
            tx,ty=tx.to(device),ty.to(device)
            vax,vay=vax.to(device),vay.to(device)
            tex,tey=tex.to(device),tey.to(device)
            ctmp=dict(base); ctmp["seed"]=s; ist=get_init_state(ctmp,device)
            seed_data[s]=(tx,ty,vax,vay,tex,tey,ist)

        bar_pos={t:i for i,t in enumerate(tasks)}

        def _run(name, s, run_idx):
            cfg=dict(merged[name]); cfg["seed"]=s
            tx,ty,vax,vay,tex,tey,ist=seed_data[s]; bp=bar_pos[(name,s)]
            task_desc = f"{name} | {mt} | ft={base['frac_train']} | seed={s}"
            _update_progress(current_run=run_idx, current_task=task_desc)
            try:
                res = OPTIMIZER_REGISTRY[name](cfg,ist,tx,ty,vax,vay,tex,tey,device,bp)
                grokked = res.grokking_step is not None
                with _PROGRESS_LOCK:
                    _PROGRESS["completed"].append({
                        "name":name, "model":mt, "split":base["frac_train"],
                        "seed":s, "grokked":grokked, "wall_time":res.wall_time,
                        "val_acc":res.final_val_acc
                    })
                if grokked:
                    _ntfy(f"✓ {name} grokked at step {res.grokking_step} ({res.grokking_wall:.1f}s) | {mt} ft={base['frac_train']} seed={s}",
                           title="Grokked!", tags="white_check_mark")
                return (name, res)
            except Exception as e:
                import traceback; print(f"\n  ✗ FAILED: {name} seed={s} — {e}"); traceback.print_exc()
                with _PROGRESS_LOCK:
                    _PROGRESS["errors"].append({"name":name, "seed":s, "error":str(e)})
                _ntfy(f"⚠ {name} FAILED: {e} | {mt} seed={s}", title="Error", priority="high", tags="warning")
                # [A4-H3] return a CRASH STUB (empty curves) instead of dropping
                # the run, so the crash is counted in the summary/JSON and never
                # silently vanishes. cfg carries this run's model_type/ft/val_ratio.
                return (name, _crash_stub(name, cfg, e))

        run_idx = _PROGRESS.get("current_run", 0)
        for n,s in tasks:
            run_idx += 1
            _,r = _run(n, s, run_idx)
            if r is not None: results_by_opt[n].append(r)

    # ══════════════════════════════════════════════════════════════════
    #  Post-processing (shared)
    # ══════════════════════════════════════════════════════════════════
    total_wall=time.time()-total_t0; results_by_opt=dict(results_by_opt)
    suffix=f"_{mt}_ft{int(base['frac_train']*100)}"
    if results_by_opt:
        print_summary(results_by_opt, total_wall, mt, base["frac_train"])
        save_json(results_by_opt, save_dir, total_wall, mt, base["frac_train"])
        plot_comparison(results_by_opt, save_dir, base["early_stop_threshold"], base["frac_train"], mt, suffix)
        print(f"Plots saved → {save_dir}/")
    return results_by_opt

# ── Multi-Split / Architecture / Full Sweep Runners ──────────────────
def run_multi_split(splits, **kwargs):
    all_results={}
    for ft in splits:
        print(f"\n{'#'*70}\n  SPLIT: {ft*100:.0f}/{(1-ft)*100:.0f}  (train/test)\n{'#'*70}")
        all_results[ft] = run_pipeline(frac_train=ft, **kwargs)
    if all_results: _plot_split_comparison(all_results, kwargs.get("save_dir","results"), kwargs.get("model_type","decoder"))
    return all_results

def run_architecture_comparison(model_types=None, **kwargs):
    if model_types is None: model_types=["decoder","vit","mamba"]
    all_results={}
    for mt in model_types:
        print(f"\n{'#'*70}\n  ARCHITECTURE: {MODEL_LABELS.get(mt,mt)}\n{'#'*70}")
        all_results[mt]=run_pipeline(model_type=mt, **kwargs)
    if all_results: _plot_architecture_comparison(all_results, kwargs.get("save_dir","results"), kwargs.get("frac_train",0.25))
    return all_results

def run_scale_comparison(scales=None, **kwargs):
    if scales is None: scales = ["small", "medium", "large"]
    all_results = {}
    for scale in scales:
        print(f"\n{'#'*70}\n  SCALE: {scale} — {MODEL_SCALES[scale]}\n{'#'*70}")
        all_results[scale] = run_pipeline(model_scale=scale, **kwargs)
    return all_results

def run_full_sweep(splits=None, model_types=None, **kwargs):
    if splits is None: splits=[0.10,0.25,0.50,0.80]
    if model_types is None: model_types=["decoder","vit","mamba"]
    all_results={}
    for mt in model_types:
        for ft in splits:
            print(f"\n{'#'*70}\n  {MODEL_LABELS.get(mt,mt)} | split={ft*100:.0f}/{(1-ft)*100:.0f}\n{'#'*70}")
            all_results[(ft,mt)]=run_pipeline(model_type=mt, frac_train=ft, **kwargs)
    if all_results: _plot_full_sweep(all_results, kwargs.get("save_dir","results"), splits, model_types)
    return all_results

# ── Cross-comparison plots ────────────────────────────────────────────
def _plot_split_comparison(all_results, save_dir, model_type):
    os.makedirs(save_dir, exist_ok=True)
    splits=sorted(all_results.keys()); all_opts=[]
    for ft in splits:
        for name in all_results[ft]:
            if name not in all_opts: all_opts.append(name)
    ns_,no=len(splits),len(all_opts); sc=plt.cm.viridis(np.linspace(0.2,0.9,ns_)); bw=0.8/ns_; x=np.arange(no)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,8))
    fig.suptitle(f"Grokking vs. Split — {MODEL_LABELS.get(model_type,model_type)}", fontsize=14, fontweight="bold")
    for si,ft in enumerate(splits):
        rbo=all_results[ft]; walls=[]; stps=[]
        for name in all_opts:
            runs=_live(rbo[name]) if name in rbo else []  # [A4-H3] crash stubs excluded
            if runs:
                walls.append(np.mean([r.grokking_wall or r.wall_time for r in runs]))
                stps.append(np.mean([r.grokking_step or r.total_steps for r in runs]))
            else: walls.append(0); stps.append(0)
        off=(si-ns_/2+0.5)*bw
        ax1.bar(x+off, walls, bw*0.9, color=sc[si], label=f"{ft*100:.0f}/{(1-ft)*100:.0f}", edgecolor="black", alpha=0.85)
        ax2.bar(x+off, stps, bw*0.9, color=sc[si], label=f"{ft*100:.0f}/{(1-ft)*100:.0f}", edgecolor="black", alpha=0.85)
    for ax,yl,t in [(ax1,"Wall-Clock (s)","⏱ Time to Grok"),(ax2,"Steps","Steps to Grok")]:
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY_NAMES.get(n,n) for n in all_opts],rotation=45,ha="right"); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(title="Train/Test"); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"split_comparison_{model_type}.png"),dpi=150,bbox_inches="tight"); plt.close()
    fig3,ax3=plt.subplots(figsize=(14,6)); gm=np.zeros((no,ns_))
    for si,ft in enumerate(splits):
        rbo=all_results[ft]
        for oi,name in enumerate(all_opts):
            if name in rbo: runs=rbo[name]; gm[oi,si]=np.nan_to_num(_grok_rate(runs))  # [A4-H3] live-only, NaN→0
    im=ax3.imshow(gm,cmap="RdYlGn",aspect="auto",vmin=0,vmax=1)
    ax3.set_xticks(range(ns_)); ax3.set_xticklabels([f"{ft*100:.0f}/{(1-ft)*100:.0f}" for ft in splits])
    ax3.set_yticks(range(no)); ax3.set_yticklabels([DISPLAY_NAMES.get(n,n) for n in all_opts])
    ax3.set_xlabel("Train/Test Split"); ax3.set_ylabel("Optimizer")
    ax3.set_title(f"Grok Success Rate — {MODEL_LABELS.get(model_type,model_type)}")
    for oi in range(no):
        for si in range(ns_): ax3.text(si,oi,f"{gm[oi,si]:.0%}",ha="center",va="center",fontsize=9,color="white" if gm[oi,si]<0.5 else "black")
    plt.colorbar(im,ax=ax3,label="Grok Rate"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f"split_heatmap_{model_type}.png"),dpi=150,bbox_inches="tight"); plt.close()

def _plot_architecture_comparison(all_results, save_dir, frac_train):
    os.makedirs(save_dir, exist_ok=True)
    mts=sorted(all_results.keys()); all_opts=[]
    for mt in mts:
        for name in all_results[mt]:
            if name not in all_opts: all_opts.append(name)
    nm,no=len(mts),len(all_opts); mc=[MODEL_COLORS.get(mt,"#888") for mt in mts]; bw=0.8/nm; x=np.arange(no)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,8))
    fig.suptitle(f"Architecture Comparison | split={frac_train*100:.0f}/{(1-frac_train)*100:.0f}", fontsize=14, fontweight="bold")
    for mi,mt in enumerate(mts):
        rbo=all_results[mt]; walls=[]; stps=[]
        for name in all_opts:
            runs=_live(rbo[name]) if name in rbo else []  # [A4-H3] crash stubs excluded
            if runs:
                walls.append(np.mean([r.grokking_wall or r.wall_time for r in runs]))
                stps.append(np.mean([r.grokking_step or r.total_steps for r in runs]))
            else: walls.append(0); stps.append(0)
        off=(mi-nm/2+0.5)*bw
        ax1.bar(x+off, walls, bw*0.9, color=mc[mi], label=MODEL_LABELS.get(mt,mt), edgecolor="black", alpha=0.85)
        ax2.bar(x+off, stps, bw*0.9, color=mc[mi], label=MODEL_LABELS.get(mt,mt), edgecolor="black", alpha=0.85)
    for ax,yl,t in [(ax1,"Wall-Clock (s)","⏱ Time to Grok"),(ax2,"Steps","Steps to Grok")]:
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY_NAMES.get(n,n) for n in all_opts],rotation=45,ha="right"); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(title="Arch"); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,"architecture_comparison.png"),dpi=150,bbox_inches="tight"); plt.close()
    fig2,axes=plt.subplots(1,nm,figsize=(7*nm,5),sharey=True)
    if nm==1: axes=[axes]
    fig2.suptitle("Val Accuracy Curves by Architecture", fontsize=13, fontweight="bold")
    for mi,mt in enumerate(mts):
        ax=axes[mi]; rbo=all_results[mt]
        for name,runs in rbo.items():
            clr=COLORS.get(name,"#888888"); dname=DISPLAY_NAMES.get(name,name)
            steps, mean, std = _interpolate_runs(runs, "val_accs")
            if len(steps) == 0: continue
            mean_smooth = np.array(_ema_smooth(list(mean), 0.92))
            ax.plot(steps, mean_smooth, label=dname, color=clr, linewidth=1.5)
            if len(runs) > 1:
                ax.fill_between(steps, np.clip(mean_smooth-std,-0.05,None), mean_smooth+std, color=clr, alpha=0.1)
        ax.set_xlabel("Steps"); ax.set_ylabel("Val Acc")
        ax.set_title(f"{MODEL_LABELS.get(mt,mt)}\n{TASK_LABELS.get(mt,'')}"); ax.set_ylim(-0.05,1.05)
        ax.axhline(y=0.95,color="red",ls="--",alpha=0.5); ax.legend(fontsize=6,ncol=2); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,"architecture_val_curves.png"),dpi=150,bbox_inches="tight"); plt.close()
    fig3,ax3=plt.subplots(figsize=(10,6)); gm=np.zeros((no,nm))
    for mi,mt in enumerate(mts):
        rbo=all_results[mt]
        for oi,name in enumerate(all_opts):
            if name in rbo: runs=rbo[name]; gm[oi,mi]=np.nan_to_num(_grok_rate(runs))  # [A4-H3] live-only, NaN→0
    im=ax3.imshow(gm,cmap="RdYlGn",aspect="auto",vmin=0,vmax=1)
    ax3.set_xticks(range(nm)); ax3.set_xticklabels([MODEL_LABELS.get(mt,mt) for mt in mts],fontsize=8)
    ax3.set_yticks(range(no)); ax3.set_yticklabels([DISPLAY_NAMES.get(n,n) for n in all_opts])
    ax3.set_title("Grok Success Rate — Architecture × Optimizer")
    for oi in range(no):
        for mi in range(nm): ax3.text(mi,oi,f"{gm[oi,mi]:.0%}",ha="center",va="center",fontsize=10,color="white" if gm[oi,mi]<0.5 else "black")
    plt.colorbar(im,ax=ax3,label="Grok Rate"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"architecture_heatmap.png"),dpi=150,bbox_inches="tight"); plt.close()

def _plot_full_sweep(all_results, save_dir, splits, model_types):
    os.makedirs(save_dir, exist_ok=True)
    all_opts=[]
    for rbo in all_results.values():
        for name in rbo:
            if name not in all_opts: all_opts.append(name)
    ns_,nm,no=len(splits),len(model_types),len(all_opts); nc=ns_*nm
    gm=np.full((no,nc),np.nan); cl=[]
    for si,ft in enumerate(splits):
        for mi,mt in enumerate(model_types):
            ci=si*nm+mi; cl.append(f"{ft*100:.0f}%|{mt[:3].upper()}")
            key=(ft,mt)
            if key in all_results:
                rbo=all_results[key]
                for oi,name in enumerate(all_opts):
                    if name in rbo: runs=rbo[name]; gm[oi,ci]=_grok_rate(runs)  # [A4-H3] live-only (NaN if all crashed; nan_to_num at imshow)
    fig,ax=plt.subplots(figsize=(max(14,nc*1.2),max(6,no*0.5)))
    im=ax.imshow(np.nan_to_num(gm,nan=0),cmap="RdYlGn",aspect="auto",vmin=0,vmax=1)
    ax.set_xticks(range(nc)); ax.set_xticklabels(cl,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(no)); ax.set_yticklabels([DISPLAY_NAMES.get(n,n) for n in all_opts])
    ax.set_title("Full Sweep — Grok Success Rate\nRows: Optimizers | Cols: Split × Architecture", fontsize=12, fontweight="bold")
    for si in range(1,ns_): ax.axvline(x=si*nm-0.5,color="white",linewidth=2)
    for oi in range(no):
        for ci in range(nc):
            v=gm[oi,ci]
            if not np.isnan(v): ax.text(ci,oi,f"{v:.0%}",ha="center",va="center",fontsize=7,color="white" if v<0.5 else "black")
    plt.colorbar(im,ax=ax,label="Grok Rate",shrink=0.8); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"full_sweep_heatmap.png"),dpi=150,bbox_inches="tight"); plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  ██████   ██    ██  ███    ██
#  ██   ██  ██    ██  ████   ██
#  ██████   ██    ██  ██ ██  ██
#  ██   ██  ██    ██  ██  ██ ██
#  ██   ██   ██████   ██   ████
#
#  Just change MODE below. No commenting/uncommenting needed.
# ═══════════════════════════════════════════════════════════════════════

# Per-optimizer hyperparameter base configs (module-level so the offline
# tuner in tuning/ can import them; __main__ uses the same object).
OPTIMIZER_CONFIGS = {
    "adamw":      {"weight_decay": 1.0},
    "neuralgrok": {"weight_decay": 1.0, "neural_alpha": 10.0, "neural_beta": 4.0,
                   # neural_layers=2 matches the CUDA kernel's documented contract
                   # (neuralgrok.py get_weights: "C++ kernel only evaluates a
                   # 2-layer MLP ... Set num_layers=2 for exact CUDA/Python
                   # parity"). The old value 3 silently trained a 3-layer net
                   # while the kernel applied an incoherent first+last sandwich.
                   #
                   # neural_hidden=16 is a CONSERVATIVE parity choice that makes
                   # the trained net == the deployed net under EITHER reading of
                   # the kernel. Evidence: the only neuralgrok_psi_forward
                   # instantiation visible in the tree is the MEGA path's
                   # <kPsiHidden=16> (csrc/fused/sm_90/opt_components.cuh:61/225,
                   # "Matches the per-op neuralgrok kernel's default
                   # instantiation"; mega cells pack exactly 3*kPsiHidden+1
                   # floats). The path step() actually binds —
                   # launch_fused_neuralgrok_full_step (bindings.cpp:1085/1108)
                   # — receives hidden_dim as a RUNTIME arg (:1110) but has NO
                   # in-tree definition, so whether it is pinned to 16 or loops
                   # over runtime hidden_dim is NOT source-confirmable here (and
                   # with no-builds, not empirically either). 16 sidesteps the
                   # question: if pinned-16, it matches the pin; if runtime, we
                   # now pass 16. The old 128 was safe ONLY under the runtime
                   # reading — under a 16-pin it silently truncated 128→16, the
                   # same train/deploy split as the A3 defect (one layer down),
                   # which is the asymmetry that makes 16 the safe choice now
                   # that the A3 amplifier-training fix is LIVE. NOTE: this
                   # parity is CONFIG-enforced, not class-enforced — the
                   # NeuralGrok ctor default stays hidden_dim=128, so direct
                   # construction outside this config still builds a 128-wide net.
                   "neural_layers": 2, "neural_hidden": 16, "inner_steps": 1},
    "grokadamw":  {"weight_decay": 1.0, "grokadamw_alpha": 0.98, "grokadamw_lamb": 2.0,
                   # PUBLISHED GrokAdamW (cognitivecomputations/grokadamw): gamma
                   # is the LAYER-WISE β1 decay (β1_i = β1*(1-gamma)**i) and kappa
                   # is grokking_signal_decay_rate (α schedule). Both WIRED now —
                   # gamma drives per-tensor β1 in the binding; kappa drives the
                   # grokking-signal α in step(). Reference defaults: gamma 0.1,
                   # kappa 0.1, lamb 2.0.
                   "grokadamw_gamma": 0.1, "grokadamw_kappa": 0.1,
                   "grokadamw_alpha_update_freq": 50,
                   "grokadamw_grad_clip": 1.0},
    "supergrok":  {"weight_decay": 1.0, "supergrok_alpha": 0.98, "supergrok_lamb": 1.0,  # identity default: lamb is now a live multiplier (lamb=1 ⇒ prior validated (1-gate)*alpha behavior); the tuner explores the dial.
                   "supergrok_gamma": 0.1, "supergrok_kappa": 0.1, "supergrok_warmup": 100,
                   "supergrok_warmup_ramp": 100, "supergrok_grad_clip": 1.0,
                   "supergrok_meta_dim": 32, "supergrok_gate_temp": 5.0,
                   "supergrok_alpha_update_freq": 50, "supergrok_meta_update_freq": 5,
                   "supergrok_zero_loss_thresh": 1e-4, "supergrok_zero_acc_thresh": 0.995},
                   # (the meta_gate_power suppression ratchet was REMOVED from the
                   # SuperGroks entirely — owner no-suppression directive; the
                   # two-term lookahead meta objective replaces it)
    "supergrok15":{"weight_decay": 1.0, "supergrok15_alpha": 0.98, "supergrok15_lamb": 2.0,
                   "supergrok15_gamma": 0.1, "supergrok15_kappa": 0.1, "supergrok15_warmup": 100,
                   "supergrok15_warmup_ramp": 100, "supergrok15_grad_clip": 1.0,
                   "supergrok15_meta_dim": 32, "supergrok15_alpha_update_freq": 50,
                   "supergrok15_zero_loss_thresh": 1e-4, "supergrok15_zero_acc_thresh": 0.995,
                   "supergrok15_sam_rho": 0.05,
                   "supergrok15_gate_scale": 20.0, "supergrok15_gate_thresh": 0.8,
                   "supergrok15_sam_freq_min": 3, "supergrok15_sam_freq_max": 20,
                   "supergrok15_sam_scale": 20.0, "supergrok15_sam_thresh": 0.85,
                   "supergrok15_bilevel_freq_min": 5, "supergrok15_bilevel_freq_max": 30,
                   "supergrok15_bilevel_scale": 20.0, "supergrok15_bilevel_thresh": 0.9,
                   "supergrok15_wd_ramp": 4.0, "supergrok15_wd_scale": 20.0,
                   "supergrok15_wd_thresh": 0.9},
    "supergrok2": {"weight_decay": 1.0, "sg2_alpha": 0.98, "sg2_lamb": 2.0,
                   "sg2_gamma": 0.1, "sg2_kappa": 0.1, "sg2_warmup": 100,
                   "sg2_warmup_ramp": 100, "sg2_grad_clip": 1.0,
                   # Key-name truth fix (audit): the four architecture entries
                   # here previously used names train_supergrok2 never reads, so
                   # they were silently IGNORED and SG2 always ran constructor
                   # defaults. Renamed to the real keys with the values that
                   # actually ran (d_model=8, 144 experts, gru_hidden=4) to keep
                   # continuity with every validated SG2 result; raising them
                   # (e.g. 1024 experts, the old intent) is a deliberate
                   # architecture change for the owner to opt into.
                   "sg2_d_model": 8, "sg2_num_experts": 144,
                   "sg2_expert_hidden": 16, "sg2_gru_hidden": 4,
                   "sg2_num_peer_heads": 4, "sg2_meta_rescale": 0.1,
                   "sg2_alpha_update_freq": 50,
                   "sg2_zero_loss_thresh": 1e-4, "sg2_zero_acc_thresh": 0.995,
                   "sg2_sam_rho": 0.05,
                   "sg2_gate_scale": 20.0, "sg2_gate_thresh": 0.8,
                   "sg2_sam_freq_min": 3, "sg2_sam_freq_max": 20,
                   "sg2_sam_scale": 20.0, "sg2_sam_thresh": 0.85,
                   "sg2_bilevel_freq_min": 5, "sg2_bilevel_freq_max": 30,
                   "sg2_bilevel_scale": 20.0, "sg2_bilevel_thresh": 0.9,
                   "sg2_wd_ramp": 4.0, "sg2_wd_scale": 20.0,
                   "sg2_wd_thresh": 0.9, "sg2_sam_enable_threshold": 0.0},
    "grokfast":   {"weight_decay": 1.0, "grokfast_alpha": 0.98, "grokfast_lamb": 2.0},
    "muon":       {"weight_decay": 1.0, "muon_lr": 0.02, "muon_momentum": 0.95},
    "lion":       {"lion_lr": 3e-4, "lion_wd": 3.0},
    "looksam":    {"weight_decay": 1.0, "looksam_rho": 0.05, "looksam_k": 5,
                   "looksam_alpha": 0.7},
    "prodigy":    {"weight_decay": 1.0, "prodigy_lr": 1.0},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grokking Race — GCP VM Edition (Multi-GPU)")
    parser.add_argument("--setup", action="store_true", help="Install dependencies and exit")
    parser.add_argument("--ntfy", type=str, default=None, metavar="TOPIC",
                        help="ntfy.sh topic for phone notifications (e.g. my-grok-run)")
    parser.add_argument("--port", type=int, default=8080, help="Status server port (default: 8080)")
    parser.add_argument("--no-status-server", action="store_true", help="Disable HTTP status server")
    parser.add_argument("--gpus", type=str, default=None, metavar="IDS",
                        help="Comma-separated GPU IDs for multi-GPU (e.g. 0,1,2,3). "
                             "Auto-detects all GPUs if set to 'auto'. "
                             "Single GPU for fair sequential benchmark if omitted.")
    parser.add_argument("--grad-hooks", action="store_true",
                        help="Use gradient hooks for L2-warm optimizer updates")
    parser.add_argument("--no-fused", action="store_true",
                        help="Disable fused (model, optimizer, arch) dispatch kernels")
    parser.add_argument("--val-ratio", type=float, default=None,
                        help="Fraction of train portion carved out as val (default: 0.10, auto 0.05 on 10/90)")
    parser.add_argument("--early-stop-test-acc", type=float, default=0.95,
                        help="Test accuracy threshold for early stopping (default: 0.95)")
    parser.add_argument("--early-stop-max-steps", type=int, default=20000,
                        help="Max steps before forced stop (default: 20000)")
    parser.add_argument("--eval-every", type=int, default=10,
                        # [A4-H1] default 100→10 (owner: track metrics every 10
                        # gradient steps). The old argparse default 100 silently
                        # overrode DEFAULT_CONFIG's eval_every=10 (argparse always
                        # sets DEFAULT_CONFIG["eval_every"] below), so the race
                        # only evaluated every 100 steps despite the directive.
                        help="Evaluate val accuracy every N steps (default: 10)")
    parser.add_argument("--optimizers", type=str, default=None,
                        help="Comma-separated optimizer names to run (default: all 11)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: mode-dependent)")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Number of seeds to use from default seed list")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated model types: decoder,vit,mamba")
    parser.add_argument("--train-test-ratios", type=str, default=None,
                        help="Comma-separated train/test ratios e.g. '10/90,25/75,50/50,80/20'")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    args = parser.parse_args()

    if args.setup:
        run_setup()

    DEFAULT_CONFIG["use_grad_hooks"] = args.grad_hooks if hasattr(args, 'grad_hooks') else False
    if args.no_fused:
        DEFAULT_CONFIG["use_fused"] = False
    DEFAULT_CONFIG["early_stop_threshold"] = args.early_stop_test_acc
    DEFAULT_CONFIG["early_stop_max_steps"] = args.early_stop_max_steps
    DEFAULT_CONFIG["eval_every"] = args.eval_every
    if args.val_ratio is not None:
        DEFAULT_CONFIG["val_ratio"] = args.val_ratio

    warnings.filterwarnings('ignore')

    # ── Parse GPU IDs ──────────────────────────────────────────────────
    gpu_ids = None
    if args.gpus is not None:
        import torch as _torch_check
        n_avail = _torch_check.cuda.device_count() if _torch_check.cuda.is_available() else 0
        if args.gpus.lower() == "auto":
            gpu_ids = list(range(n_avail)) if n_avail >= 2 else None
        else:
            try:
                gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
                gpu_ids = [g for g in gpu_ids if g < n_avail]
            except ValueError:
                print(f"  ⚠ Invalid --gpus value: {args.gpus!r}")
                gpu_ids = None
        if gpu_ids and len(gpu_ids) >= 2:
            print(f"\n  ▸ Multi-GPU enabled: {len(gpu_ids)} GPUs {gpu_ids}")
            for g in gpu_ids:
                print(f"    GPU {g}: {_torch_check.cuda.get_device_name(g)}")
        elif gpu_ids and len(gpu_ids) == 1:
            print(f"\n  ▸ Single GPU: cuda:{gpu_ids[0]} ({_torch_check.cuda.get_device_name(gpu_ids[0])})")
        else:
            print(f"\n  ▸ No valid GPUs found for multi-GPU. Falling back to default.")
            gpu_ids = None

    # ── Notifications ─────────────────────────────────────────────────
    if args.ntfy:
        _NTFY_TOPIC = args.ntfy
        _ntfy("🚀 Grokking race starting!", title="Race Started", tags="rocket")
        _start_ntfy_listener()

    # ── Status server ─────────────────────────────────────────────────
    if not args.no_status_server:
        start_status_server(args.port)

    # ┌─────────────────────────────────────────────────────────────┐
    # │  CHANGE THIS to switch what runs:                           │
    # │    "A" — Single run  (1 model, 1 split)                    │
    # │    "B" — Multi-split  (1 model, 4 splits: 10/25/50/80)     │
    # │    "C" — Architecture comparison  (Decoder vs ViT vs Mamba) │
    # │    "D" — Full sweep  (all splits × all architectures)       │
    # │    "E" — Scale comparison  (small/medium/large models)     │
    # └─────────────────────────────────────────────────────────────┘
    MODE = "D"

    # ── Per-optimizer hyperparameters ─────────────────────────────────
    optimizer_configs = dict(OPTIMIZER_CONFIGS)  # see module-level OPTIMIZER_CONFIGS

    ALL_OPTIMIZERS = ["adamw","neuralgrok","grokadamw","supergrok","supergrok15",
                      "supergrok2","grokfast","muon","lion","looksam","prodigy"]

    _common = dict(
        optimizer_configs=optimizer_configs,
        compile_model=True,
        parallel=False,
        max_steps=args.early_stop_max_steps,
        lr=1e-3,
        threshold=args.early_stop_test_acc,
        # [A4-M8] log_every removed — it was never read by any train loop
        # (write-only into config); dropped from DEFAULT_CONFIG, _common, and
        # run_pipeline's signature/merge.
        save_dir=args.output,
        gpu_ids=gpu_ids,
        use_amp=False,
        val_ratio=args.val_ratio,
        early_stop_max_steps=args.early_stop_max_steps,
        eval_every=args.eval_every,
    )

    # ── Parse CLI overrides for optimizers, seeds, tasks, splits ──────
    if args.optimizers:
        ALL_OPTIMIZERS = [o.strip().lower() for o in args.optimizers.split(",")]
    if args.train_test_ratios:
        SPLITS = []
        for r in args.train_test_ratios.split(","):
            parts = r.strip().split("/")
            SPLITS.append(int(parts[0]) / 100.0)
    else:
        SPLITS = [0.10, 0.25, 0.50, 0.80]

    SEEDS_A = [42, 123, 456, 1337, 3407, 9999]
    SEEDS_BCD = [42, 123, 456, 1337, 3407]
    if args.seeds:
        custom_seeds = [int(s.strip()) for s in args.seeds.split(",")]
        SEEDS_A = custom_seeds; SEEDS_BCD = custom_seeds
    elif args.num_seeds:
        SEEDS_A = SEEDS_A[:args.num_seeds]; SEEDS_BCD = SEEDS_BCD[:args.num_seeds]

    if args.tasks:
        ARCHS = [t.strip().lower() for t in args.tasks.split(",")]
    else:
        ARCHS = ["decoder", "vit", "mamba"]

    if MODE == "A":   total = len(ALL_OPTIMIZERS) * len(SEEDS_A)
    elif MODE == "B": total = len(ALL_OPTIMIZERS) * len(SEEDS_BCD) * len(SPLITS)
    elif MODE == "C": total = len(ALL_OPTIMIZERS) * len(SEEDS_BCD) * len(ARCHS)
    elif MODE == "D": total = len(ALL_OPTIMIZERS) * len(SEEDS_BCD) * len(SPLITS) * len(ARCHS)
    elif MODE == "E": total = len(ALL_OPTIMIZERS) * len(SEEDS_BCD) * 3  # 3 scales
    else: total = 0

    _update_progress(status="running", mode=MODE, started_at=time.time(),
                     total_runs=total, current_run=0)
    gpu_msg = f" | {len(gpu_ids)} GPUs" if gpu_ids and len(gpu_ids) >= 2 else ""
    _ntfy(f"Mode {MODE} | {total} total runs{gpu_msg}", title="Config", tags="gear")

    # ── Run ───────────────────────────────────────────────────────────
    if gpu_ids and len(gpu_ids) >= 2:
        mp.set_start_method("spawn", force=True)
    race_t0 = time.time()

    if MODE == "A":
        results = run_pipeline(optimizers=ALL_OPTIMIZERS, seeds=SEEDS_A,
                               frac_train=0.25, model_type="decoder", **_common)
    elif MODE == "B":
        results = run_multi_split(splits=SPLITS, optimizers=ALL_OPTIMIZERS,
                                  seeds=SEEDS_BCD, model_type="decoder", **_common)
    elif MODE == "C":
        results = run_architecture_comparison(model_types=ARCHS, optimizers=ALL_OPTIMIZERS,
                                              seeds=SEEDS_BCD, frac_train=0.25, **_common)
    elif MODE == "D":
        results = run_full_sweep(splits=SPLITS, model_types=ARCHS, optimizers=ALL_OPTIMIZERS,
                                 seeds=SEEDS_BCD, **_common)
    elif MODE == "E":
        results = run_scale_comparison(
            scales=["small", "medium", "large"],
            optimizers=ALL_OPTIMIZERS,
            seeds=SEEDS_BCD,
            frac_train=0.25,
            model_type="decoder",
            **_common,
        )
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Use 'A', 'B', 'C', 'D', or 'E'.")

    total_time = time.time() - race_t0
    _update_progress(status="complete")

    # ── Final notification ────────────────────────────────────────────
    snap = _progress_snapshot()
    done = snap["completed"]; grokked = sum(1 for d in done if d.get("grokked"))
    final_msg = (
        f"🏁 Mode {MODE} complete!\n"
        f"Total time: {timedelta(seconds=int(total_time))}\n"
        f"Runs: {len(done)}/{total} | Grokked: {grokked} | Errors: {len(snap['errors'])}\n"
        f"Results saved to ./results/"
    )
    print(f"\n{final_msg}")
    _ntfy(final_msg, title="🏁 Race Complete!", priority="high", tags="checkered_flag")

    # Keep status server alive briefly for final queries
    if not args.no_status_server:
        print(f"\nStatus server still running on port {args.port}. Ctrl+C to exit.")
        try:
            while True: time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down.")
