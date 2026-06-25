"""Layer-A dataset-source interface for grokking_race_v2 (DEFAULT-OFF).

Importable seam for the scaled-dataset regime. `make_source_for_task(c, seed)` returns
the SAME 6-tuple the legacy `make_data_for_task` returns —
`(train_view, train_y, val_probe, val_y, test_probe, test_y)` — as plain CPU tensors:

  * slots 0,1  train_view / train_y : a FIXED, once-materialized minibatch view of a
                STREAMING corpus, sized to the step budget. The harness consumes it
                full-batch (the "<1 epoch re-see is safe" regime); the per-step stream
                is exposed via `DatasetSource.stream(step)` for callers that want true
                streaming, but the harness contract is the fixed view (zero train-loop
                churn, kernels size-pinned).
  * slots 2,3  val_probe / val_y   : a FIXED eval probe, materialized ONCE, reused every
                eval (capped forward in evaluate()).
  * slots 4,5  test_probe / test_y : a FIXED eval probe (independent draw).

The three `_*_stub` builders below are DETERMINISTIC SYNTHETIC reference sources shaped
to the CURRENT size-pinned kernels (vocab/seq/patch/class unchanged), so the harness
runs TODAY with no downloads. They are the replaceable seam for the real Phase-4 loaders
(FineWeb-Edu / ImageNet-1k / GiftEvalPretrain), which drop in behind the identical
`(c, seed) -> 6-tuple` signature. Layer-B (real vocab/patch/class + kernel regen) is OUT
OF SCOPE here.

DEFAULT-OFF: nothing in this module is imported unless c["data_source"] != "modular".
"""
import torch

# Per-step / per-probe RNG decorrelation salts (large primes).
_SEED_MUL = 1_000_003
_STEP_MUL = 100_003


def _gen(seed, step, salt):
    """Deterministic CPU generator keyed by (seed, step, salt) → A/A/A reproducible."""
    g = torch.Generator(device="cpu")
    g.manual_seed((int(seed) * _SEED_MUL + int(step) * _STEP_MUL + int(salt)) & 0x7FFFFFFFFFFF)
    return g


class DatasetSource:
    """A single-pass stream + a fixed materialized train view + fixed eval probes.

    `gen(step, bs, g)` is a per-model closure returning one fresh (x[bs,...], y[bs])
    minibatch seeded by step (no epoch boundary, deterministic). `stream(step)` exposes
    it; `train_view` is the once-collapsed finite view the harness consumes full-batch.
    """
    def __init__(self, gen, bs, n_view, vprobe, tprobe, seed):
        self._gen = gen
        self.bs = bs
        self.seed = seed
        # Materialize the fixed train view ONCE (n_view rows), deterministic.
        xs, ys = [], []
        got = 0
        step = 0
        while got < n_view:
            xb, yb = gen(step, bs, _gen(seed, step, 1))
            xs.append(xb); ys.append(yb); got += xb.shape[0]; step += 1
        x = torch.cat(xs, 0)[:n_view]
        y = torch.cat(ys, 0)[:n_view]
        self.train_view = (x, y)
        self.vprobe = vprobe       # (x, y) fixed
        self.tprobe = tprobe       # (x, y) fixed

    def stream(self, step):
        """One fresh per-step minibatch (for callers wanting true streaming)."""
        return self._gen(step, self.bs, _gen(self.seed, step, 1))

    def as_tuple(self):
        tx, ty = self.train_view
        vx, vy = self.vprobe
        ex, ey = self.tprobe
        return tx, ty, vx, vy, ex, ey


# ── Reference stub builders (deterministic synthetic; shaped to current kernels) ──
def _lm_stub(c, seed, bs):
    """FineWeb-Edu / decoder seam. seq=4 (current decoder kSeq), vocab=num_tokens."""
    seq = 4
    vocab = c.get("num_tokens", 99)

    def gen(step, b, g):
        x = torch.randint(0, vocab, (b, seq), generator=g)
        y = torch.randint(0, vocab, (b,), generator=g)
        return x, y
    n_probe = max(bs, min(c.get("eval_probe_rows", 4096), 50_000))
    gp = _gen(seed, -1, 7); vx = torch.randint(0, vocab, (n_probe, seq), generator=gp); vy = torch.randint(0, vocab, (n_probe,), generator=gp)
    gp = _gen(seed, -2, 7); ex = torch.randint(0, vocab, (n_probe, seq), generator=gp); ey = torch.randint(0, vocab, (n_probe,), generator=gp)
    return gen, (vx, vy), (ex, ey)


def _forecast_stub(c, seed, bs):
    """GiftEvalPretrain / mamba seam. seq=seq_len (current mamba=8), vocab=num_tokens."""
    seq = c.get("seq_len", 8)
    vocab = c.get("num_tokens", 99)
    classes = c.get("p", 97)

    def gen(step, b, g):
        x = torch.randint(0, vocab, (b, seq), generator=g)
        y = torch.randint(0, classes, (b,), generator=g)
        return x, y
    n_probe = max(bs, min(c.get("eval_probe_rows", 4096), 50_000))
    gp = _gen(seed, -1, 9); vx = torch.randint(0, vocab, (n_probe, seq), generator=gp); vy = torch.randint(0, classes, (n_probe,), generator=gp)
    gp = _gen(seed, -2, 9); ex = torch.randint(0, vocab, (n_probe, seq), generator=gp); ey = torch.randint(0, classes, (n_probe,), generator=gp)
    return gen, (vx, vy), (ex, ey)


def _imagenet_stub(c, seed, bs):
    """ImageNet-1k / vit seam. patches/dim = current vit layout; classes = p."""
    npatch = c.get("num_patches", 16)
    pdim = c.get("patch_dim", 49)
    classes = c.get("p", 97)

    def gen(step, b, g):
        x = torch.randn(b, npatch, pdim, generator=g)
        y = torch.randint(0, classes, (b,), generator=g)
        return x, y
    n_probe = max(bs, min(c.get("eval_probe_rows", 4096), 50_000))
    gp = _gen(seed, -1, 11); vx = torch.randn(n_probe, npatch, pdim, generator=gp); vy = torch.randint(0, classes, (n_probe,), generator=gp)
    gp = _gen(seed, -2, 11); ex = torch.randn(n_probe, npatch, pdim, generator=gp); ey = torch.randint(0, classes, (n_probe,), generator=gp)
    return gen, (vx, vy), (ex, ey)


_STUBS = {
    # data_source value : builder
    "fineweb_edu": _lm_stub,
    "gifteval":    _forecast_stub,
    "imagenet1k":  _imagenet_stub,
    # generic synthetic alias routed by model_type (no download):
    "synthetic":   None,
}


def _route(c):
    """Pick a stub builder. Explicit data_source names win; 'synthetic' routes by
    model_type so a single generic flag exercises every cell."""
    ds = c.get("data_source", "modular")
    if ds in _STUBS and _STUBS[ds] is not None:
        return _STUBS[ds]
    mt = c.get("model_type", "decoder")
    if mt == "decoder": return _lm_stub
    if mt == "mamba":   return _forecast_stub
    if mt == "vit":     return _imagenet_stub
    raise ValueError(f"dataset_sources: no stub for model_type={mt!r} / data_source={ds!r}")


def make_source_for_task(c, seed):
    """Return the legacy 6-tuple `(tx, ty, vax, vay, tex, tey)` as plain CPU tensors:
    a fixed once-materialized streaming train VIEW + a fixed eval PROBE (val/test).

    Drop-in for make_data_for_task's non-'modular' branch. The harness consumes these
    exactly like the mod-97 tensors (.to(device), full-batch train, capped evaluate())."""
    bs = int(c.get("train_batch_size", 512))
    # Fixed train-view size: a "<1 epoch" budget tied to the step budget, capped.
    budget = int(c.get("early_stop_max_steps", c.get("max_steps", 20_000)))
    n_view = int(c.get("train_view_rows", min(max(bs, budget), 16_384)))
    gen, vprobe, tprobe = _route(c)(c, seed, bs)
    src = DatasetSource(gen, bs, n_view, vprobe, tprobe, seed)
    return src.as_tuple()
