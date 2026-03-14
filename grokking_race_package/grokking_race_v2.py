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
  #   2. Subscribe to your topic (e.g. "peter-grok-2025")
  python grokking_race.py --ntfy peter-grok-2025

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

MODEL_SCALES = {
    "small":  {"dim_model": 128, "num_heads": 4, "num_layers": 2},   # ~420K params
    "medium": {"dim_model": 256, "num_heads": 8, "num_layers": 4},   # ~3.5M params
    "large":  {"dim_model": 512, "num_heads": 8, "num_layers": 6},   # ~20M params
}

DEFAULT_CONFIG: Dict = {
    "p": 97, "operation": "x/y", "frac_train": 0.5,
    "num_layers": 2, "dim_model": 128, "num_heads": 4, "num_tokens": 99,
    "lr": 1e-3, "weight_decay": 1.0, "beta1": 0.9, "beta2": 0.98,
    "max_steps": 100_000, "early_stop_threshold": 0.95,
    "early_stop_patience": 500, "log_every": 10, "seed": 42,
    "compile_model": False, "use_amp": False, "model_type": "decoder",
    "patch_dim": 49, "num_patches": 16,
    "chain_length": 3, "seq_len": 8,
}

# ── Data 1: Modular Division (a ÷ b) mod p  [Decoder] ────────────────
def make_data(p=97, frac_train=0.5, seed=42):
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
    n = int(len(pairs) * frac_train)
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return x[:n], y[:n], x[n:], y[n:]

# ── Data 2: MNIST-Addition (a + b) mod p  [ViT] ──────────────────────
def make_mnist_addition_data(p=97, frac_train=0.5, seed=42):
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
    n_split = int(len(pairs) * frac_train)
    images = []
    for a, b in pairs:
        full = torch.cat([number_images[a], number_images[b]], dim=0)
        patches = full.unfold(0, 7, 7).unfold(1, 7, 7).contiguous().reshape(16, 49)
        images.append(patches)
    x = torch.stack(images); y = torch.tensor(labels, dtype=torch.long)
    return x[:n_split], y[:n_split], x[n_split:], y[n_split:]

# ── Data 3: Sequential Chained Division  [Mamba] ─────────────────────
def make_sequential_division_data(p=97, chain_length=3, frac_train=0.5, seed=42):
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
    n = int(len(pairs) * frac_train)
    x = torch.tensor(pairs, dtype=torch.long); y = torch.tensor(labels, dtype=torch.long)
    return x[:n], y[:n], x[n:], y[n:]

def make_data_for_task(c, seed):
    mt = c.get("model_type", "decoder"); ft, p = c.get("frac_train", 0.5), c.get("p", 97)
    if mt == "decoder":  return make_data(p, ft, seed)
    elif mt == "vit":    return make_mnist_addition_data(p, ft, seed)
    elif mt == "mamba":  return make_sequential_division_data(p, c.get("chain_length", 3), ft, seed)
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
    def __init__(self, nl=2, d=128, h=4, ntok=99, seq=4):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(seq, d)
        self.layers = nn.ModuleList([DecoderBlock(d, h, seq_len=seq) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, ntok)
        self.register_buffer('pos_ids', torch.arange(seq).unsqueeze(0))
    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        for l in self.layers: h = l(h)
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
    def __init__(self, p=97, patch_dim=49, num_patches=16, d=128, h=4, nl=2):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Embedding(num_patches + 1, d)
        self.layers = nn.ModuleList([EncoderBlock(d, h) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, p)
        self.register_buffer('pos_ids', torch.arange(num_patches + 1).unsqueeze(0))
    def forward(self, x):
        B = x.size(0); h = self.patch_proj(x)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = h + self.pos(self.pos_ids)
        for l in self.layers: h = l(h)
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
    def __init__(self, p=97, ntok=99, seq_len=8, d=128, nl=2):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(seq_len, d)
        self.layers = nn.ModuleList([SelectiveSSMLayer(d) for _ in range(nl)])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, p)
        self.register_buffer('pos_ids', torch.arange(seq_len).unsqueeze(0))
    def forward(self, x):
        h = self.tok(x) + self.pos(self.pos_ids)
        for l in self.layers: h = l(h)
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
    if mt == "decoder": return Transformer(nl, d, h, c["num_tokens"], 4).to(device)
    elif mt == "vit":   return ViT(p=p, patch_dim=c.get("patch_dim",49), num_patches=c.get("num_patches",16), d=d, h=h, nl=nl).to(device)
    elif mt == "mamba": return MambaModel(p=p, ntok=c["num_tokens"], seq_len=c.get("seq_len",8), d=d, nl=nl).to(device)
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
    def __init__(self, threshold=0.95, patience=500, max_steps=100_000):
        self.threshold=threshold; self.patience=patience; self.max_steps=max_steps
        self._triggered=False; self._counter=0; self.best_val_acc=0.
        self.grokking_step=None; self.grokking_wall=None; self._t0=time.time()
    def step(self, val_acc, current_step):
        if current_step >= self.max_steps: return True
        self.best_val_acc = max(self.best_val_acc, val_acc)
        if val_acc >= self.threshold:
            if not self._triggered:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                self._triggered=True; self.grokking_step=current_step
                self.grokking_wall = time.time()-self._t0
            self._counter += 1; return self._counter >= self.patience
        else: self._counter=0; return False


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
                 "val_losses","val_accs","wall_time","total_steps",
                 "grokking_step","grokking_wall","final_val_acc","final_train_acc",
                 "model_type","frac_train")
    def __init__(self, name, seed=42, model_type="decoder", frac_train=0.5):
        self.name=name; self.seed=seed; self.model_type=model_type; self.frac_train=frac_train
        self.steps=[]; self.train_losses=[]; self.train_accs=[]
        self.val_losses=[]; self.val_accs=[]
        self.wall_time=0.; self.total_steps=0; self.grokking_step=None
        self.grokking_wall=None; self.final_val_acc=0.; self.final_train_acc=0.

def _merge(base, ov):
    m = dict(base)
    if ov: m.update(ov)
    return m
def _stopper(c):
    return EarlyStopper(c["early_stop_threshold"], c["early_stop_patience"], c["max_steps"])
def _pbar(name, mx, pos):
    return tqdm(range(1, mx+1), desc=f"{name:<14s}", position=pos, leave=True, ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]")
def _progressive_eval_freq(step, base_freq=10, max_freq=50, scale=0.01, thresh=500):
    """Sigmoid-driven eval frequency: eval less often early, more often later."""
    heat = 1.0 / (1.0 + math.exp(-scale * (step - thresh)))
    freq = max_freq - (max_freq - base_freq) * heat
    return max(base_freq, round(freq))
def _eval_log(r, step, m, tx, ty, vx, vy, c, st, pb):
    tl, ta = evaluate(m, tx, ty, c["p"]); vl, va = evaluate(m, vx, vy, c["p"])
    r.steps.append(step); r.train_losses.append(tl); r.train_accs.append(ta)
    r.val_losses.append(vl); r.val_accs.append(va)
    pb.set_postfix({"trn":f"{ta:.3f}","val":f"{va:.3f}","tl":f"{tl:.3f}","vl":f"{vl:.3f}"}, refresh=False)
    return st.step(va, step), tl, vl
def _fin(r, st, step, t0):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    r.wall_time=time.time()-t0; r.total_steps=step
    r.grokking_step=st.grokking_step; r.grokking_wall=st.grokking_wall
    r.final_train_acc = r.train_accs[-1] if r.train_accs else 0.
    r.final_val_acc = r.val_accs[-1] if r.val_accs else 0.
    return r
def _load(c, device, init_state):
    m = build_model(c, device, c.get("compile_model", False))
    try: m.load_state_dict(copy.deepcopy(init_state), strict=True)
    except RuntimeError:
        raw = m._orig_mod if hasattr(m, "_orig_mod") else m
        raw.load_state_dict(copy.deepcopy(init_state), strict=True)
    return m
def _tr(name, c):
    return TrainResult(name, c["seed"], c.get("model_type","decoder"), c.get("frac_train",0.5))

# ── C++/CUDA fused optimizers (grokking_optimizers package) ────────────
from grokking_optimizers import (
    SuperGrok15, SuperGrok2, SuperGrok11, ISABPEERMetaNet,
    GrokAdamW, NeuralGrok, Prodigy, Grokfast, Lion, LookSAM, Muon,
    CUDAGraphOptimizer,
)

def _maybe_wrap_cuda_graph(opt, c):
    """Wrap optimizer in CUDAGraphOptimizer if enabled in config."""
    if c.get("use_cuda_graph", False):
        return CUDAGraphOptimizer(
            opt,
            warmup_steps=c.get("cuda_graph_warmup", 3),
            max_graph_age=c.get("cuda_graph_max_age", 0),
        )
    return opt

# ── 1. AdamW ──────────────────────────────────────────────────────────
def train_adamw(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("AdamW",c); m=_load(c,dev,init)
    opt=torch.optim.AdamW(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]), weight_decay=c["weight_decay"])
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("AdamW",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 2. NeuralGrok ─────────────────────────────────────────────────────
# NOTE: The NeuralGrok amplifier MLP is trained here via aopt on the outer
# split, but opt.step() calls the fused C++ kernel which uses a *snapshot*
# of the amplifier weights (cached at build time).  The amplifier's
# learned weights therefore lag behind by one step.  This is intentional:
# the C++ kernel cannot call back into Python autograd, so the amplifier
# must be trained separately and its updated weights are picked up on the
# next opt.step() call when the cache is refreshed.
def train_neuralgrok(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("NeuralGrok",c); m=_load(c,dev,init)
    opt=NeuralGrok(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha=c.get("neural_alpha",10.0),
        beta=c.get("neural_beta",4.0), num_layers=c.get("neural_layers",3),
        hidden_dim=c.get("neural_hidden",128), inner_steps=c.get("inner_steps",1),
        grad_clip=c.get("neural_grad_clip",1.0))
    opt.amplifier=opt.amplifier.to(dev)
    aopt=opt.get_amplifier_optimizer(lr=1e-3)
    ni=int(tx.size(0)*0.9); ix,ox,iy,oy = tx[:ni],tx[ni:],ty[:ni],ty[ni:]
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("NeuralGrok",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(ix),iy)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        aopt.zero_grad()
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            aloss=c.get("neural_beta",4.0)*F.cross_entropy(m(ox),oy)
        scaler.scale(aloss).backward(); scaler.step(aopt); scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 3. GrokAdamW ──────────────────────────────────────────────────────
def train_grokadamw(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("GrokAdamW",c); m=_load(c,dev,init)
    opt=GrokAdamW(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha=c.get("grokadamw_alpha",0.98),
        lamb=c.get("grokadamw_lamb",5.0), gamma=c.get("grokadamw_gamma",0.1),
        decay=c.get("grokadamw_decay",0.1), grad_clip=c.get("grokadamw_grad_clip",1.0))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("GrokAdamW",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 4. SuperGrok v1.1 ─────────────────────────────────────────────────
def train_supergrok(c, init, tx, ty, vx, vy, dev, bp=0):
    muf=c.get("supergrok_meta_update_freq",5)
    r=_tr("SuperGrok",c); m=_load(c,dev,init)
    opt=SuperGrok11(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], alpha_init=c.get("supergrok_alpha",0.98),
        lamb=c.get("supergrok_lamb",5.0), gamma=c.get("supergrok_gamma",0.1),
        kappa=c.get("supergrok_kappa",0.1), warmup_steps=c.get("supergrok_warmup",100),
        warmup_ramp=c.get("supergrok_warmup_ramp",100),
        gradient_clipping=c.get("supergrok_grad_clip",1.0),
        alpha_update_freq=c.get("supergrok_alpha_update_freq",50),
        gate_temperature=c.get("supergrok_gate_temp",5.0),
        zero_loss_threshold=c.get("supergrok_zero_loss_thresh",1e-4),
        zero_acc_threshold=c.get("supergrok_zero_acc_thresh",0.995),
        meta_hidden_dim=c.get("supergrok_meta_dim",32))
    opt.meta_net=opt.meta_net.to(dev)
    mopt=torch.optim.Adam(opt.meta_net.parameters(), lr=1e-4)
    crit_sg=nn.CrossEntropyLoss()
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("SuperGrok",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            logits=m(tx); loss=F.cross_entropy(logits,ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        # Check for inf/nan gradients from AMP unscaling
        _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
        if _has_inf:
            scaler.update(); continue
        train_loss_val=loss.item()
        with torch.no_grad():
            train_acc=(logits.detach()[:,:c["p"]].argmax(-1)==ty).float().mean().item()
        if step%muf==0:
            try: opt.meta_step(m, vx, vy, crit_sg, mopt)
            except Exception as e: warnings.warn(f"SuperGrok meta_step failed at step {step}: {e}")
        # SAM step (v1.1 sharpness-aware minimization)
        sam_freq = max(1, muf * 2)
        if hasattr(opt, 'sam_step') and step % sam_freq == 0:
            try: opt.sam_step(m, tx, ty, crit_sg)
            except Exception as e: warnings.warn(f"SuperGrok sam_step failed at step {step}: {e}")
        kw={"train_loss":train_loss_val, "train_acc":train_acc}
        if step%c.get("supergrok_alpha_update_freq",50)==0:
            with torch.no_grad():
                vl_sg=F.cross_entropy(m(vx),vy).item()
            kw["val_loss"]=vl_sg
        try: opt.step(**kw)
        except TypeError: opt.step()
        scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 4b. SuperGrok v1.5 ────────────────────────────────────────────────
def train_supergrok15(c, init, tx, ty, vx, vy, dev, bp=0):
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
        wd_thresh=c.get("supergrok15_wd_thresh",0.9))
    opt.meta_net=opt.meta_net.to(dev)
    mopt=torch.optim.Adam(opt.meta_net.parameters(), lr=1e-4)
    crit_s15=nn.CrossEntropyLoss()
    alpha_freq=c.get("supergrok15_alpha_update_freq",50)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("SuperGrok1.5",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            logits=m(tx); loss=F.cross_entropy(logits,ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        # Check for inf/nan gradients from AMP unscaling
        _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
        if _has_inf:
            scaler.update(); continue
        # Adaptive SAM (sigmoid-driven frequency)
        sam_freq_eff=opt._get_effective_sam_freq()
        if step%sam_freq_eff==0:
            try: opt.sam_step(m, tx, ty, crit_s15)
            except Exception as e: warnings.warn(f"SuperGrok1.5 sam_step failed at step {step}: {e}")
        # Adaptive bilevel (independent sigmoid-driven frequency)
        bilevel_freq_eff=opt._get_effective_bilevel_freq()
        if step%bilevel_freq_eff==0:
            try: opt.bilevel_step(m, tx, ty, vx, vy, crit_s15, mopt)
            except Exception as e: warnings.warn(f"SuperGrok1.5 bilevel_step failed at step {step}: {e}")
        # Deferred metrics — only compute .item() when needed
        eval_freq=_progressive_eval_freq(step, base_freq=c["log_every"])
        kw={}
        needs_metrics=(step%alpha_freq==0) or (step%eval_freq==0) or step==1
        if needs_metrics:
            with torch.no_grad():
                train_loss_val=loss.item()
                train_acc=(logits.detach()[:,:c["p"]].argmax(-1)==ty).float().mean().item()
            kw["train_loss"]=train_loss_val; kw["train_acc"]=train_acc
            if step%alpha_freq==0:
                with torch.no_grad():
                    vl_s15=F.cross_entropy(m(vx),vy).item()
                kw["val_loss"]=vl_s15
        try: opt.step(**kw)
        except TypeError: opt.step()
        scaler.update()
        if step%eval_freq==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 4c. SuperGrok v2 (Mamba-3 + PEER) ────────────────────────────────
def train_supergrok2(c, init, tx, ty, vx, vy, dev, bp=0):
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
        sam_enable_threshold=c.get("sg2_sam_enable_threshold",0.0))
    opt.meta_net=opt.meta_net.to(dev)
    mopt=torch.optim.Adam(opt.meta_net.parameters(), lr=1e-4)
    crit_s2=nn.CrossEntropyLoss()
    alpha_freq=c.get("sg2_alpha_update_freq",50)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("SuperGrok2",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            logits=m(tx); loss=F.cross_entropy(logits,ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        # Check for inf/nan gradients from AMP unscaling
        _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
        if _has_inf:
            scaler.update(); continue
        # Adaptive SAM (sigmoid-driven frequency)
        sam_freq_eff=opt._get_effective_sam_freq()
        if step%sam_freq_eff==0:
            try: opt.sam_step(m, tx, ty, crit_s2)
            except Exception as e: warnings.warn(f"SuperGrok2 sam_step failed at step {step}: {e}")
        # Adaptive bilevel (independent sigmoid-driven frequency)
        bilevel_freq_eff=opt._get_effective_bilevel_freq()
        if step%bilevel_freq_eff==0:
            try: opt.bilevel_step(m, tx, ty, vx, vy, crit_s2, mopt)
            except Exception as e: warnings.warn(f"SuperGrok2 bilevel_step failed at step {step}: {e}")
        # Deferred metrics
        eval_freq=_progressive_eval_freq(step, base_freq=c["log_every"])
        kw={}
        needs_metrics=(step%alpha_freq==0) or (step%eval_freq==0) or step==1
        if needs_metrics:
            with torch.no_grad():
                train_loss_val=loss.item()
                train_acc=(logits.detach()[:,:c["p"]].argmax(-1)==ty).float().mean().item()
            kw["train_loss"]=train_loss_val; kw["train_acc"]=train_acc
            if step%alpha_freq==0:
                with torch.no_grad():
                    vl_s2=F.cross_entropy(m(vx),vy).item()
                kw["val_loss"]=vl_s2
        try: opt.step(**kw)
        except TypeError: opt.step()
        scaler.update()
        if step%eval_freq==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 5. Grokfast ───────────────────────────────────────────────────────
def train_grokfast(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("Grokfast",c); m=_load(c,dev,init)
    opt=Grokfast(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], grokfast_alpha=c.get("grokfast_alpha",0.98),
        grokfast_lamb=c.get("grokfast_lamb",2.0))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("Grokfast",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 6. Muon ───────────────────────────────────────────────────────────
def train_muon(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("Muon",c); m=_load(c,dev,init)
    muon_params, adam_params = [], []
    for n,p in m.named_parameters():
        if p.requires_grad: (muon_params if p.ndim==2 else adam_params).append(p)
    opt=Muon(muon_params, params_1d=adam_params if adam_params else None,
        lr=c.get("muon_lr",0.02), momentum=c.get("muon_momentum",0.95),
        weight_decay=c["weight_decay"], adamw_lr=c["lr"],
        adamw_betas=(c["beta1"],c["beta2"]))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("Muon",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 7. Lion ───────────────────────────────────────────────────────────
def train_lion(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("Lion",c); m=_load(c,dev,init)
    opt=Lion(m.parameters(), lr=c.get("lion_lr",3e-4),
        betas=(c["beta1"],0.99), weight_decay=c.get("lion_wd",3.0))
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("Lion",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 9. LookSAM ───────────────────────────────────────────────────────
def train_looksam(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("LookSAM",c); m=_load(c,dev,init)
    k=c.get("looksam_k",5)
    opt=LookSAM(m.parameters(), lr=c["lr"], betas=(c["beta1"],c["beta2"]),
        weight_decay=c["weight_decay"], rho=c.get("looksam_rho",0.05),
        k=k, alpha=c.get("looksam_alpha",0.7))
    crit_ls=nn.CrossEntropyLoss()
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("LookSAM",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        # Check for inf/nan gradients from AMP unscaling
        _has_inf = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters())
        if _has_inf:
            scaler.update(); continue
        if opt.should_sam_step():
            opt.sam_step(m, tx, ty, crit_ls)
        opt.step()
        scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

# ── 10. Prodigy ───────────────────────────────────────────────────────
def train_prodigy(c, init, tx, ty, vx, vy, dev, bp=0):
    r=_tr("Prodigy",c); m=_load(c,dev,init)
    opt=Prodigy(m.parameters(), lr=c.get("prodigy_lr",1.0), weight_decay=c["weight_decay"])
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
    st=_stopper(c); m.train(); t0=time.time()
    for step in (pb:=_pbar("Prodigy",c["max_steps"],bp)):
        with torch.amp.autocast('cuda', enabled=c.get("use_amp",False)):
            loss=F.cross_entropy(m(tx),ty)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step%c["log_every"]==0 or step==1:
            done,_,_=_eval_log(r,step,m,tx,ty,vx,vy,c,st,pb)
            if done: break
    pb.close(); return _fin(r,st,step,t0)

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

def _interpolate_runs(runs, attr, num_points=500):
    """Interpolate multiple runs onto a common step grid, return mean ± std."""
    if not runs: return [], [], []
    max_step = max(r.steps[-1] for r in runs if r.steps)
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
                 f"train/val={ft*100:.0f}/{(1-ft)*100:.0f} | {thresh*100:.0f}% threshold",
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
    ns = sorted(rbo.keys(), key=lambda n: np.mean([r.grokking_wall or r.wall_time for r in rbo[n]]))
    fig2,(ax1,ax2) = plt.subplots(1,2,figsize=(18,6))
    fig2.suptitle(f"Grokking Race — {mt_label}  [{task_label}] | split={ft*100:.0f}/{(1-ft)*100:.0f}", fontsize=13, fontweight="bold")
    for i,name in enumerate(ns):
        runs=rbo[name]; wt=[r.grokking_wall or r.wall_time for r in runs]; gs=[r.grokking_step or r.total_steps for r in runs]
        clr=COLORS.get(name,"#888888"); dname=DISPLAY_NAMES.get(name,name)
        nogrok=any(r.grokking_wall is None for r in runs)
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
    ranked=sorted(rbo.items(), key=lambda kv: np.mean([r.grokking_wall or 1e9 for r in kv[1]]))
    multi=any(len(v)>1 for v in rbo.values())
    hdr=f"  {'#':>2} {'Optimizer':<14} {'Grok Wall (s)':>14} {'Grok Steps':>12} {'Total Steps':>12} {'Val Acc':>9} {'Status':>8}"
    if multi: hdr+=f" {'Seeds':>6}"
    print(hdr); print("  "+"-"*(w-2))
    medals=["🥇","🥈","🥉"]+["  "]*20
    for rank,(name,runs) in enumerate(ranked):
        dname=DISPLAY_NAMES.get(name,name)
        gw=[r.grokking_wall or r.wall_time for r in runs]; va=[r.final_val_acc for r in runs]
        nogrok=any(r.grokking_wall is None for r in runs)
        line=(f"  {medals[rank]} {dname:<14} {np.mean(gw):>14.2f} "
              f"{np.mean([r.grokking_step or r.total_steps for r in runs]):>12,.0f} "
              f"{np.mean([r.total_steps for r in runs]):>12,.0f} "
              f"{np.mean(va):>9.4f} {'✗ DNF' if nogrok else '✓ GROK':>8}")
        if multi: line+=f" {len(runs):>6}"
        print(line)
    print("  "+"-"*(w-2))
    if total_wall: print(f"  Pipeline wall: {total_wall:.1f}s"); print("="*w)

def save_json(rbo, save_dir="results", total_wall=None, model_type="decoder", frac_train=0.5):
    os.makedirs(save_dir, exist_ok=True)
    d={"_meta":{"total_wall":total_wall,"model_type":model_type,"frac_train":frac_train}}
    for name,runs in rbo.items():
        d[name]=[{"seed":r.seed,"steps":r.steps,"train_losses":r.train_losses,"train_accs":r.train_accs,
            "val_losses":r.val_losses,"val_accs":r.val_accs,"wall_time":r.wall_time,"total_steps":r.total_steps,
            "grokking_step":r.grokking_step,"grokking_wall":r.grokking_wall,
            "final_val_acc":r.final_val_acc,"final_train_acc":r.final_train_acc,
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

            # Lazily create data for this seed on this GPU
            if s not in seed_data:
                torch.manual_seed(s); np.random.seed(s)
                tx, ty, vx, vy = make_data_for_task(base, s)
                tx, ty = tx.to(device), ty.to(device)
                vx, vy = vx.to(device), vy.to(device)
                ctmp = dict(base); ctmp["seed"] = s
                ist = get_init_state(ctmp, device)
                seed_data[s] = (tx, ty, vx, vy, ist)

            cfg = dict(merged[name]); cfg["seed"] = s
            tx, ty, vx, vy, ist = seed_data[s]
            try:
                res = OPTIMIZER_REGISTRY[name](cfg, ist, tx, ty, vx, vy, device, 0)
                result_queue.put((name, s, res))
                grokked = res.grokking_step is not None
                status = f"✓ grokked step {res.grokking_step}" if grokked else f"✗ DNF"
                print(f"  [GPU {gpu_id}] {DISPLAY_NAMES.get(name,name)} seed={s}: {status} ({res.wall_time:.1f}s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [GPU {gpu_id}] ✗ {name} seed={s} FAILED: {e}")
                result_queue.put((name, s, None))

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  [GPU {gpu_id}] Worker {worker_id} crashed: {e}")


# ── run_pipeline ──────────────────────────────────────────────────────
def run_pipeline(optimizers=None, optimizer_configs=None, seeds=None,
                 compile_model=False, parallel=True, max_steps=None,
                 lr=None, weight_decay=None, threshold=None, log_every=None,
                 frac_train=None, seed=None, device_str=None,
                 save_dir="results", model_type=None, gpu_ids=None,
                 use_amp=False, model_scale=None):
    base=dict(DEFAULT_CONFIG)
    for k,v in [("max_steps",max_steps),("lr",lr),("weight_decay",weight_decay),
                ("early_stop_threshold",threshold),("log_every",log_every),
                ("frac_train",frac_train),("seed",seed),("model_type",model_type)]:
        if v is not None: base[k]=v
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
    print(f"\n{'='*60}\n  Model : {mt_label}\n  Task  : {TASK_LABELS.get(mt,'')}\n"
          f"  Device: {dev_str}\n  Split : {base['frac_train']*100:.0f}/{(1-base['frac_train'])*100:.0f}\n"
          f"  Seeds : {seeds}\n  Max   : {base['max_steps']:,} steps\n{'='*60}")
    merged={n: _merge(base, optimizer_configs.get(n)) for n in optimizers}
    for n in merged: merged[n]["model_type"]=mt; merged[n]["frac_train"]=base["frac_train"]; merged[n]["seq_len"]=base["seq_len"]
    tasks=[(n,s) for n in optimizers for s in seeds]
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    # Print data/param info using first device
    tmp_dev = device if not use_multi_gpu else torch.device(f"cuda:{gpu_ids[0]}")
    torch.manual_seed(seeds[0]); np.random.seed(seeds[0])
    tx0,ty0,vx0,vy0 = make_data_for_task(base, seeds[0])
    print(f"Train: {tx0.shape[0]:,} | Val: {vx0.shape[0]:,} | x shape: {list(tx0.shape)}")
    npar=sum(p.numel() for p in _raw_model(base,tmp_dev).parameters())
    print(f"Params ({mt}): {npar:,}")
    del tx0, ty0, vx0, vy0

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
                if res is not None:
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
                    with _PROGRESS_LOCK:
                        _PROGRESS["errors"].append({"name":name, "seed":s, "error":"worker returned None"})
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
            tx,ty,vx,vy=make_data_for_task(base, s)
            tx,ty=tx.to(device),ty.to(device); vx,vy=vx.to(device),vy.to(device)
            ctmp=dict(base); ctmp["seed"]=s; ist=get_init_state(ctmp,device)
            seed_data[s]=(tx,ty,vx,vy,ist)

        bar_pos={t:i for i,t in enumerate(tasks)}

        def _run(name, s, run_idx):
            cfg=dict(merged[name]); cfg["seed"]=s
            tx,ty,vx,vy,ist=seed_data[s]; bp=bar_pos[(name,s)]
            task_desc = f"{name} | {mt} | ft={base['frac_train']} | seed={s}"
            _update_progress(current_run=run_idx, current_task=task_desc)
            try:
                res = OPTIMIZER_REGISTRY[name](cfg,ist,tx,ty,vx,vy,device,bp)
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
                return (name, None)

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
        print(f"\n{'#'*70}\n  SPLIT: {ft*100:.0f}/{(1-ft)*100:.0f}  (train/val)\n{'#'*70}")
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
            if name in rbo:
                runs=rbo[name]; walls.append(np.mean([r.grokking_wall or r.wall_time for r in runs]))
                stps.append(np.mean([r.grokking_step or r.total_steps for r in runs]))
            else: walls.append(0); stps.append(0)
        off=(si-ns_/2+0.5)*bw
        ax1.bar(x+off, walls, bw*0.9, color=sc[si], label=f"{ft*100:.0f}/{(1-ft)*100:.0f}", edgecolor="black", alpha=0.85)
        ax2.bar(x+off, stps, bw*0.9, color=sc[si], label=f"{ft*100:.0f}/{(1-ft)*100:.0f}", edgecolor="black", alpha=0.85)
    for ax,yl,t in [(ax1,"Wall-Clock (s)","⏱ Time to Grok"),(ax2,"Steps","Steps to Grok")]:
        ax.set_xticks(x); ax.set_xticklabels([DISPLAY_NAMES.get(n,n) for n in all_opts],rotation=45,ha="right"); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(title="Train/Val"); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"split_comparison_{model_type}.png"),dpi=150,bbox_inches="tight"); plt.close()
    fig3,ax3=plt.subplots(figsize=(14,6)); gm=np.zeros((no,ns_))
    for si,ft in enumerate(splits):
        rbo=all_results[ft]
        for oi,name in enumerate(all_opts):
            if name in rbo: runs=rbo[name]; gm[oi,si]=sum(1 for r in runs if r.grokking_step)/len(runs)
    im=ax3.imshow(gm,cmap="RdYlGn",aspect="auto",vmin=0,vmax=1)
    ax3.set_xticks(range(ns_)); ax3.set_xticklabels([f"{ft*100:.0f}/{(1-ft)*100:.0f}" for ft in splits])
    ax3.set_yticks(range(no)); ax3.set_yticklabels([DISPLAY_NAMES.get(n,n) for n in all_opts])
    ax3.set_xlabel("Train/Val Split"); ax3.set_ylabel("Optimizer")
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
            if name in rbo:
                runs=rbo[name]; walls.append(np.mean([r.grokking_wall or r.wall_time for r in runs]))
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
            if name in rbo: runs=rbo[name]; gm[oi,mi]=sum(1 for r in runs if r.grokking_step)/len(runs)
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
                    if name in rbo: runs=rbo[name]; gm[oi,ci]=sum(1 for r in runs if r.grokking_step)/len(runs)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grokking Race — GCP VM Edition (Multi-GPU)")
    parser.add_argument("--setup", action="store_true", help="Install dependencies and exit")
    parser.add_argument("--ntfy", type=str, default=None, metavar="TOPIC",
                        help="ntfy.sh topic for phone notifications (e.g. peter-grok-2025)")
    parser.add_argument("--port", type=int, default=8080, help="Status server port (default: 8080)")
    parser.add_argument("--no-status-server", action="store_true", help="Disable HTTP status server")
    parser.add_argument("--gpus", type=str, default=None, metavar="IDS",
                        help="Comma-separated GPU IDs for multi-GPU (e.g. 0,1,2,3). "
                             "Auto-detects all GPUs if set to 'auto'. "
                             "Single GPU for fair sequential benchmark if omitted.")
    args = parser.parse_args()

    if args.setup:
        run_setup()

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
    optimizer_configs = {
        "adamw":      {"weight_decay": 1.0},
        "neuralgrok": {"weight_decay": 1.0, "neural_alpha": 10.0, "neural_beta": 4.0,
                       "neural_layers": 3, "neural_hidden": 128, "inner_steps": 1},
        "grokadamw":  {"weight_decay": 1.0, "grokadamw_alpha": 0.98, "grokadamw_lamb": 5.0,
                       "grokadamw_gamma": 0.1, "grokadamw_decay": 0.1, "grokadamw_grad_clip": 1.0},
        "supergrok":  {"weight_decay": 1.0, "supergrok_alpha": 0.98, "supergrok_lamb": 5.0,
                       "supergrok_gamma": 0.1, "supergrok_kappa": 0.1, "supergrok_warmup": 100,
                       "supergrok_warmup_ramp": 100, "supergrok_grad_clip": 1.0,
                       "supergrok_meta_dim": 32, "supergrok_gate_temp": 5.0,
                       "supergrok_alpha_update_freq": 50, "supergrok_meta_update_freq": 5,
                       "supergrok_zero_loss_thresh": 1e-4, "supergrok_zero_acc_thresh": 0.995},
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
                       "sg2_num_inducing": 16, "sg2_meta_d_model": 8,
                       "sg2_num_peer_experts": 1024, "sg2_expert_hidden": 4,
                       "sg2_recurrent_dim": 8, "sg2_meta_rescale": 0.1,
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

    ALL_OPTIMIZERS = ["adamw","neuralgrok","grokadamw","supergrok","supergrok15",
                      "supergrok2","grokfast","muon","lion","looksam","prodigy"]

    _common = dict(
        optimizer_configs=optimizer_configs,
        compile_model=True,
        parallel=False,
        max_steps=20_000,
        lr=1e-3,
        threshold=0.95,
        log_every=10,
        save_dir="results",
        gpu_ids=gpu_ids,
        use_amp=False,
    )

    # ── Compute total runs for progress tracking ──────────────────────
    SEEDS_A = [42, 123, 456, 1337, 3407, 9999]
    SEEDS_BCD = [42, 123, 456, 1337, 3407]
    SPLITS = [0.10, 0.25, 0.50, 0.80]
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
