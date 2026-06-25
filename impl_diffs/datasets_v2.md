# AREA: grokking_race_v2.py — Layer-A data plumbing + dead-code removal (APPLY-READY, v2)

Repo: `/workspace/SuperGrok1.5`. Single target file: `grokking_race_v2.py`, plus ONE
new module `grokking_optimizers/dataset_sources.py` (importable stub interface, no
downloads). The lead applies the OLD→NEW edits below with exact-match replacement.

> **v2 status (re-validated against the live file, 2026-06-25).** Every OLD snippet
> below was copied VERBATIM from the current `grokking_race_v2.py` (2505 lines) and the
> live line numbers re-checked (they had NOT drifted from the v1 spec, except where
> noted). The four behavioral changes vs v1 `datasets.md`:
>
> 1. **Edit 2.9b rewritten as 6 exact-match OLD→NEW blocks** (v1 told the lead to "apply
>    by line number bottom-up" for the 5 shared call sites because the 2-line context was
>    not unique). Verified: the line **two above** each `opt=_maybe_wrap_cuda_graph(opt, c)`
>    call differs at every site, so a 3-line OLD block is unique at all 6 sites. No
>    line-number bookkeeping needed.
> 2. **Edit 2.9a line range corrected** to the verbatim 8-line block at `:893-900`
>    (def body is `:895-898`; the OLD/NEW match is unchanged from v1, only the cited range
>    label is fixed).
> 3. Confirmed the dead-code KEEP: `g.MambaModel(...)` is imported and instantiated by
>    `tests/hw/test_mamba_megakernel.py:37-39` (`import grokking_race_v2 as g`), and the
>    `mamba_oracle.py` / `mamba3_oracle.py` transcribe `SelectiveSSMLayer`. Both classes
>    are LIVE — DO NOT remove. `_maybe_checkpoint` is consumed by `MambaModel.forward`
>    (`:488`), `Transformer.forward` (`:402`), `ViT.forward` (`:430`) — LIVE, KEEP.
> 4. Confirmed `evaluate()` has exactly ONE real caller (`_fin` `:815`; the other hits at
>    `:275/:719/:721` are comments) and `EarlyStopper(...)` has exactly ONE caller
>    (`_stopper` `:691`), so the new default-valued params are fully backward-compatible.

## Scope / design decision (unchanged from v1; re-affirmed)

The task pins the knob to **`c['data_source']`** and scopes this to **Layer-A data
plumbing ONLY** with the mod-97 path **BYTE-IDENTICAL** (kernels are size-pinned;
Layer-B vocab/seq/patch/class resize NOT in scope).

**Chosen design (minimal-diff, byte-identical when OFF, no train-loop churn):**

* New DEFAULT-OFF config axis `"data_source": "modular"` in `DEFAULT_CONFIG`. Every
  existing run, test, and tuner config is untouched (mod-97 byte-identical). No CLI flag
  is added, so the gate command (which passes no `--data-source`) stays `"modular"`.
* `make_data_for_task` dispatches on `data_source`:
  * `"modular"` (default) → the existing three builders, returned **VERBATIM** as the
    same 6-tuple → provably byte-identical (the literal original 4 lines).
  * any other value → import `grokking_optimizers.dataset_sources.make_source_for_task`,
    returning the SAME 6-tuple shape `(train_view, train_y, val_probe, val_y,
    test_probe, test_y)` as **plain CPU tensors**: a FIXED, once-materialized streaming
    **train view** (slots 0,1) + a **FIXED eval probe** (slots 2,3 = val, 4,5 = test).
    Because they are plain tensors, the existing `.to(device)` data-load sites, the
    full-batch train loops, and `_eval_log`/`evaluate`/`_fin` all work with **ZERO
    edits**.
* `evaluate()` gains an **inert-when-OFF** fixed-probe-capped micro-batched path so a
  large probe forwards in seconds. For mod-97 the cap is OFF (`micro_batch<=0` ⇒ the
  original single-shot lines run verbatim) → byte-identical.
* `EarlyStopper` gains a DEFAULT-OFF `mode="acc"` axis with a **loss-PLATEAU** branch for
  LM/forecasting; the `acc` branch (ViT 0.95 grok + all mod-97) is byte-identical.
* Dead-code: remove `_maybe_wrap_cuda_graph` def + its 6 identity call sites. **KEEP**
  `MambaModel`/`SelectiveSSMLayer`/`_maybe_checkpoint` (all proven LIVE above).

**Layer-B (vocab/seq/patch/class resize + kernel regen) is explicitly NOT in scope.**

---

# FILE 1 (NEW): `grokking_optimizers/dataset_sources.py`

Importable stub interface. No downloads, no heavy deps (torch only). Deterministic by
`(seed, step)` so A/A/A holds. Create the file with EXACTLY this content:

```python
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
```

> NOTE on `train_view_rows`: `make_source_for_task` reads it via
> `c.get("train_view_rows", <budget default>)`. A literal `None` in the dict would
> override the default with `None` and break `int(None)`. The dispatcher (Edit 2.2)
> therefore POPS a falsy `train_view_rows` from the copy it passes, so the module's
> `.get(..., default)` fires. (DEFAULT_CONFIG ships `"train_view_rows": None` for
> documentation; the dispatcher normalizes it.)

---

# FILE 2: `grokking_race_v2.py`

## Edit 2.1 — `DEFAULT_CONFIG`: add DEFAULT-OFF Layer-A knobs

OLD (verbatim, `:287-292`):

```python
    "early_stop_patience": 50, "seed": 42,
    "compile_model": False, "use_amp": False, "model_type": "decoder",
    "patch_dim": 49, "num_patches": 16,
    "chain_length": 3, "seq_len": 8,
    "use_fused": True,
}
```

NEW:

```python
    "early_stop_patience": 50, "seed": 42,
    "compile_model": False, "use_amp": False, "model_type": "decoder",
    "patch_dim": 49, "num_patches": 16,
    "chain_length": 3, "seq_len": 8,
    "use_fused": True,
    # ── Layer-A scaled-dataset axis (P3, DEFAULT-OFF). "modular" = the legacy
    # in-memory mod-p grokking task (full-batch GD, whole-tensor eval), byte-
    # identical to pre-knob. Any other value dispatches make_data_for_task to
    # grokking_optimizers.dataset_sources (streaming train VIEW + FIXED eval
    # probe). Layer-B (real vocab/patch/class resize + kernel regen) is OUT OF
    # SCOPE; the stub sources are shaped to the current size-pinned kernels.
    "data_source": "modular",          # modular | fineweb_edu | imagenet1k | gifteval | synthetic
    "train_batch_size": 512,           # per-step minibatch for the streaming regime
    "train_view_rows": None,           # fixed train-view size (None → derived from budget)
    "eval_probe_rows": 4096,           # FIXED eval-probe size (built once, reused)
    "eval_micro_batch": 0,             # >0 → capped micro-batched evaluate() (0=single-shot)
    # ── Early-stop mode (DEFAULT-OFF). "acc" = legacy 0.95-grok trigger (ViT +
    # every mod-p cell), byte-identical. "loss" = val/test-LOSS PLATEAU stop for
    # LM (decoder) / forecasting (mamba) where 0.95-acc is meaningless at scale.
    "early_stop_mode": "acc",          # acc | loss
    "early_stop_plateau_patience": 30, # # evals w/o loss improvement before plateau stop
    "early_stop_plateau_min_delta": 1e-4,  # min loss decrease counted as improvement
}
```

## Edit 2.2 — `make_data_for_task`: dispatch on DEFAULT-OFF `data_source`

OLD (verbatim, `:372-378`):

```python
def make_data_for_task(c, seed):
    mt = c.get("model_type", "decoder"); ft, p = c.get("frac_train", 0.5), c.get("p", 97)
    vr = c.get("val_ratio", 0.10)
    if mt == "decoder":  return make_data(p, ft, vr, seed)
    elif mt == "vit":    return make_mnist_addition_data(p, ft, vr, seed)
    elif mt == "mamba":  return make_sequential_division_data(p, c.get("chain_length", 3), ft, vr, seed)
    else: raise ValueError(f"Unknown model_type: {mt}")
```

NEW:

```python
def make_data_for_task(c, seed):
    mt = c.get("model_type", "decoder"); ft, p = c.get("frac_train", 0.5), c.get("p", 97)
    vr = c.get("val_ratio", 0.10)
    # Layer-A scaled-dataset dispatch (DEFAULT-OFF). data_source=="modular" is the
    # legacy in-memory mod-p task, returned VERBATIM (byte-identical). Any other
    # value routes to the streaming train-VIEW + FIXED eval-probe stub interface;
    # the returned 6-tuple has the identical shape/role contract, so .to(device),
    # the full-batch train loops, and evaluate() are unchanged. Layer-B resize is
    # OUT OF SCOPE (stubs are shaped to the current size-pinned kernels).
    if c.get("data_source", "modular") != "modular":
        from grokking_optimizers.dataset_sources import make_source_for_task
        cc = dict(c)
        if not cc.get("train_view_rows"):   # normalize None → module's budget default
            cc.pop("train_view_rows", None)
        return make_source_for_task(cc, seed)
    if mt == "decoder":  return make_data(p, ft, vr, seed)
    elif mt == "vit":    return make_mnist_addition_data(p, ft, vr, seed)
    elif mt == "mamba":  return make_sequential_division_data(p, c.get("chain_length", 3), ft, vr, seed)
    else: raise ValueError(f"Unknown model_type: {mt}")
```

> The `modular` branch is the **literal original 4 lines** — byte-identical execution
> for every existing run. The lazy `from ... import` inside the non-modular branch means
> `import grokking_race_v2` (the gate's import check) never touches `dataset_sources`.

## Edit 2.3 — `evaluate`: inert-when-OFF fixed-probe-capped micro-batched forward

OLD (verbatim, `:535-540`):

```python
@torch.no_grad()
def evaluate(model, x, y, p=97):
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits[:, :p].argmax(-1) == y).float().mean().item()
    return loss, acc
```

NEW:

```python
@torch.no_grad()
def evaluate(model, x, y, p=97, micro_batch=0):
    # micro_batch<=0 (mod-p default) → the original single-shot forward, byte-
    # identical. micro_batch>0 → capped micro-batched forward over a FIXED eval
    # probe so a large probe forwards in seconds, not one giant activation. The
    # sum/÷n form reproduces the single-shot mean (per-forward fp32 order
    # preserved within each chunk; cross-chunk is an fp32 add of per-chunk sums).
    n = x.shape[0]
    if micro_batch is None or micro_batch <= 0 or micro_batch >= n:
        logits = model(x)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits[:, :p].argmax(-1) == y).float().mean().item()
        return loss, acc
    tot_loss = 0.0; tot_correct = 0
    for i in range(0, n, micro_batch):
        xb = x[i:i+micro_batch]; yb = y[i:i+micro_batch]
        logits = model(xb)
        tot_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        tot_correct += (logits[:, :p].argmax(-1) == yb).sum().item()
    return tot_loss / n, tot_correct / n
```

> For mod-97 every caller passes `micro_batch=0` (or omits it) → the single-shot
> branch runs the original three lines verbatim. The `micro_batch >= n` guard also
> routes the trivial "one chunk covers everything" case to the byte-identical single
> shot. The only real caller is `_fin` `:815` (Edit 2.7a), which threads
> `getattr(_fin, "_eval_micro_batch", 0)` (defaults 0). `_eval_log` does NOT call
> `evaluate()` (it inlines its own fused forward) → untouched.

## Edit 2.4 — `EarlyStopper`: DEFAULT-OFF `mode` + loss-PLATEAU branch (acc branch unchanged)

OLD (verbatim, `:542-569`):

```python
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
```

NEW:

```python
class EarlyStopper:
    def __init__(self, threshold=0.95, max_steps=20_000, patience=500, metric_name="test_acc",
                 mode="acc", plateau_patience=None, plateau_min_delta=1e-4):
        self.threshold=threshold; self.max_steps=max_steps; self.patience=patience
        self.metric_name=metric_name  # which accuracy feeds step(): "test_acc" or "val_acc"
        # [A4-M1/M2] best_metric_acc / metric_acc are metric-agnostic names: under
        # val-stopping this tracks val, under test-stopping it tracks test — the
        # old test_acc-specific names mislabelled the val-criterion runs.
        # mode (P3, DEFAULT "acc"): "acc" → the legacy >=threshold grok trigger (ViT
        # ImageNet + every mod-p cell), BYTE-IDENTICAL. "loss" → PLATEAU stop for the
        # LM (decoder) / forecasting (mamba) scaled cells: stop after plateau_patience
        # evals with no loss improvement beyond plateau_min_delta.
        self.mode=mode
        self.plateau_patience = plateau_patience if plateau_patience is not None else patience
        self.plateau_min_delta = plateau_min_delta
        self._best_loss=float("inf"); self._plateau_counter=0
        self._triggered=False; self._counter=0; self.best_metric_acc=0.
        self.grokking_step=None; self.grokking_wall=None; self._t0=time.time()
        self.stopping_reason=None; self.stopping_step=None
    def step(self, metric_acc, current_step):
        # `metric_acc` is ACCURACY in acc-mode; in loss-mode the caller passes LOSS.
        if current_step >= self.max_steps:
            if self.stopping_reason is None:
                self.stopping_reason="max_steps"; self.stopping_step=current_step
            return True
        if self.mode == "loss":
            # PLATEAU: track best (lowest) loss; stop after plateau_patience evals with
            # no improvement beyond plateau_min_delta. grokking_step anchors at the first
            # improvement so downstream "grokked" reporting still has a step.
            signal = metric_acc
            if signal < self._best_loss - self.plateau_min_delta:
                self._best_loss = signal; self._plateau_counter = 0
                if self.grokking_step is None:
                    self.grokking_step = current_step
                    self.grokking_wall = time.time() - self._t0
            else:
                self._plateau_counter += 1
                if self._plateau_counter >= self.plateau_patience:
                    if self.stopping_reason is None:
                        self.stopping_reason=f"{self.metric_name}_plateau"; self.stopping_step=current_step
                    return True
            return False
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
```

> The `acc`-mode tail (`self.best_metric_acc = max(...)` onward) is the **literal
> original** block — identical trigger, patience, `best_metric_acc`, `grokking_step`,
> `grokking_wall`, `{metric}_threshold` reason. `mode` defaults to `"acc"`, so every
> mod-97 run is byte-identical.

## Edit 2.5 — `_stopper`: pass DEFAULT-OFF `mode` + plateau knobs

OLD (verbatim, `:687-692`):

```python
def _stopper(c):
    # early_stop_on: "test" (default, historical) or "val" — which accuracy
    # triggers the threshold/patience stop. Tuner + the val-criterion race use "val".
    metric = "val_acc" if c.get("early_stop_on", "test") == "val" else "test_acc"
    return EarlyStopper(c["early_stop_threshold"], c.get("early_stop_max_steps", c["max_steps"]),
                        c["early_stop_patience"], metric_name=metric)
```

NEW:

```python
def _stopper(c):
    # early_stop_on: "test" (default, historical) or "val" — which accuracy
    # triggers the threshold/patience stop. Tuner + the val-criterion race use "val".
    metric = "val_acc" if c.get("early_stop_on", "test") == "val" else "test_acc"
    mode = c.get("early_stop_mode", "acc")   # "acc" (ViT/mod-p) | "loss" (LM/forecast plateau)
    if mode == "loss":
        metric = "val_loss" if c.get("early_stop_on", "test") == "val" else "test_loss"
    return EarlyStopper(c["early_stop_threshold"], c.get("early_stop_max_steps", c["max_steps"]),
                        c["early_stop_patience"], metric_name=metric, mode=mode,
                        plateau_patience=c.get("early_stop_plateau_patience"),
                        plateau_min_delta=c.get("early_stop_plateau_min_delta", 1e-4))
```

> When `early_stop_mode` is absent or `"acc"` (the default), this constructs the
> EarlyStopper with `mode="acc"`, `plateau_patience=None` (→ `patience`),
> `plateau_min_delta=1e-4` — the plateau fields are inert and the `metric`/threshold
> args are identical to before. Byte-identical for mod-97. (`_stopper` is the SOLE
> `EarlyStopper(...)` caller — verified `:691`.)

## Edit 2.6 — `_eval_log`: feed LOSS to the stopper in loss-mode (acc-mode unchanged)

The full-eval tail selects what `st.step()` consumes. In `acc`-mode it must pass
accuracy (unchanged); in `loss`-mode it must pass the loss. Both the fast-val path and
the full path are edited; the `acc`-mode value of `stop_acc` is the original expression.

### 2.6a — fast-val path

OLD (verbatim, `:736-739`):

```python
        stop_acc = va_f if c.get("early_stop_on", "test") == "val" else None
        if stop_acc is None:
            return False, None, None  # fast path only valid with val stopping
        return st.step(stop_acc, step), None, None
```

NEW:

```python
        stop_acc = va_f if c.get("early_stop_on", "test") == "val" else None
        if stop_acc is None:
            return False, None, None  # fast path only valid with val stopping
        # loss-mode plateau stopper consumes the val LOSS, not the accuracy.
        if c.get("early_stop_mode", "acc") == "loss":
            stop_acc = vl_f
        return st.step(stop_acc, step), None, None
```

> `vl_f` is in scope (bound at `:735`, `vl_f, va_f = va_t.cpu().tolist()`).

### 2.6b — full-eval path

OLD (verbatim, `:757-758`):

```python
    stop_acc = va if c.get("early_stop_on", "test") == "val" else tea
    return st.step(stop_acc, step), tl, tel
```

NEW:

```python
    stop_acc = va if c.get("early_stop_on", "test") == "val" else tea
    # loss-mode plateau stopper consumes the LOSS matching early_stop_on.
    if c.get("early_stop_mode", "acc") == "loss":
        stop_acc = vl if c.get("early_stop_on", "test") == "val" else tel
    return st.step(stop_acc, step), tl, tel
```

> `vl`/`tel` are in scope (bound at `:747`,
> `(tl, ta), (vl, va), (tel, tea) = torch.stack(outs).cpu().tolist()`). In `acc`-mode
> (default), `stop_acc` is the **original expression** and the new `if` is skipped →
> byte-identical. The three full forwards in `_eval_log` are NOT capped here (mod-97
> sizes are tiny; the train view is already a small fixed `train_view_rows` tensor and
> the probe is `eval_probe_rows`-bounded). The final test eval IS capped (Edit 2.7).

## Edit 2.7 — `_fin`: capped final-test eval + plateau-aware confirm

### 2.7a — capped final eval

OLD (verbatim, `:813-816`):

```python
    m.eval()
    with torch.no_grad():
        r.final_test_loss, r.final_test_acc = evaluate(m, tex, tey, p)
    m.train()
```

NEW:

```python
    m.eval()
    with torch.no_grad():
        r.final_test_loss, r.final_test_acc = evaluate(
            m, tex, tey, p, micro_batch=getattr(_fin, "_eval_micro_batch", 0))
    m.train()
```

> `_fin` has no `c` in scope (signature is `def _fin(r, st, step, t0, m, tex, tey, p=97)`,
> verified `:803`). The micro-batch is threaded via a function attribute set at run start
> (Edit 2.8). It defaults to `0` (attribute absent → `getattr` default `0`) → single-shot,
> byte-identical for mod-97.

### 2.7b — plateau-aware `grokking_step_test_confirmed`

OLD (verbatim, `:826-830`):

```python
    if r.grokking_step is not None and r.steps and r.test_accs:
        gi = min(range(len(r.steps)), key=lambda i: abs(r.steps[i] - r.grokking_step))
        r.grokking_step_test_confirmed = bool(r.test_accs[gi] >= st.threshold - 0.05)
    else:
        r.grokking_step_test_confirmed = False
```

NEW:

```python
    if getattr(st, "mode", "acc") == "loss":
        # loss-mode: "confirmed" just means a best-loss anchor was recorded; the
        # 0.95-acc test confirmation is meaningless for LM/forecasting plateau cells.
        r.grokking_step_test_confirmed = (r.grokking_step is not None)
    elif r.grokking_step is not None and r.steps and r.test_accs:
        gi = min(range(len(r.steps)), key=lambda i: abs(r.steps[i] - r.grokking_step))
        r.grokking_step_test_confirmed = bool(r.test_accs[gi] >= st.threshold - 0.05)
    else:
        r.grokking_step_test_confirmed = False
```

> `getattr(st, "mode", "acc")` is `"acc"` for every mod-97 stopper (Edit 2.4 defaults
> `mode="acc"`), so the `elif` runs the **original branch** verbatim → byte-identical.

## Edit 2.8 — thread `eval_micro_batch` into `_fin` at run start (in `_tr`)

`_tr` (`:846-859`) is called by every `train_*` at run start and already has `c` in
scope. Set the `_fin` micro-batch attribute there so it carries to the single `_fin`
call per loop without editing 9 train loops or the `_fin` signature.

OLD (verbatim, `:858-859`):

```python
    r.use_fused_requested = bool(c.get("use_fused", True))  # task 2: guard input
    return r
```

NEW:

```python
    r.use_fused_requested = bool(c.get("use_fused", True))  # task 2: guard input
    # P3: thread the (DEFAULT-OFF) capped-eval micro-batch to _fin's final test eval
    # without widening _fin's signature. 0 (mod-p default) → single-shot, byte-identical.
    _fin._eval_micro_batch = int(c.get("eval_micro_batch", 0) or 0)
    return r
```

> `_tr` runs once per train_* invocation, single model per process (same discipline as
> the `LAST_L3_ENGINE` module global, which `_tr` also resets at `:853-854`), so the
> attribute is always set immediately before the loop's `_fin` call. For mod-97 it is
> `0` → byte-identical.

## Edit 2.9 — DEAD CODE: remove `_maybe_wrap_cuda_graph` def + 6 identity call sites

`_maybe_wrap_cuda_graph(opt, c)` returns `opt` unchanged (pure identity, no side effects,
no external refs — verified `:895-898`). Remove the def and all 6 calls.
**KEEP** `MambaModel`/`SelectiveSSMLayer`/`_maybe_checkpoint` (proven LIVE; see the v2
status note at the top — no edit to them).

> Live grep confirms exactly 7 occurrences: the def `:895` + 6 calls at
> `:983, :1090, :1387, :1424, :1452, :1512`. After applying all 7 edits below,
> `grep -n "_maybe_wrap_cuda_graph" grokking_race_v2.py` must return NOTHING.

### 2.9a — delete the definition

OLD (verbatim, `:893-900`):

```python
    return "L1+eager"                      # legacy/inert: eager/L1 path removed

def _maybe_wrap_cuda_graph(opt, c):
    """No-op shim. CUDA Graph wrapping was removed in the post-refactor
    cleanup; the race is single-node and does not need graph capture."""
    return opt

def _try_fused_train_step(model_name, opt_name, model, optimizer, x_batch,
```

NEW:

```python
    return "L1+eager"                      # legacy/inert: eager/L1 path removed

def _try_fused_train_step(model_name, opt_name, model, optimizer, x_batch,
```

### 2.9b — delete each of the 6 call sites (6 unique exact-match blocks)

The 6 call lines are textually identical (`    opt=_maybe_wrap_cuda_graph(opt, c)`). v1
told the lead to fall back to line-number bookkeeping for the 5 shared sites; v2 instead
gives a UNIQUE multi-line OLD block for each (the line TWO above each call differs at
every site — verified live). Apply all 6 as exact-match OLD→NEW; order does not matter
because each OLD block is globally unique.

**Site 1 (train_adamw, `:982-984`):**

OLD:
```python
    # AdamW baseline does not support use_grad_hooks (no _single_param_step API).
    opt=_maybe_wrap_cuda_graph(opt, c)
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
```
NEW:
```python
    # AdamW baseline does not support use_grad_hooks (no _single_param_step API).
    scaler=torch.amp.GradScaler('cuda', enabled=c.get("use_amp",False))
```

**Site 2 (train_grokadamw, `:1088-1090`):**

OLD:
```python
        lamb=c.get("grokadamw_lamb",2.0), grad_clip=c.get("grokadamw_grad_clip",1.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
```
NEW:
```python
        lamb=c.get("grokadamw_lamb",2.0), grad_clip=c.get("grokadamw_grad_clip",1.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
```

**Site 3 (train_grokfast, `:1385-1387`):**

OLD:
```python
        grokfast_lamb=c.get("grokfast_lamb",2.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
```
NEW:
```python
        grokfast_lamb=c.get("grokfast_lamb",2.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
```

**Site 4 (train_muon, `:1422-1424`):**

OLD:
```python
        adamw_betas=(c["beta1"],c["beta2"]),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
```
NEW:
```python
        adamw_betas=(c["beta1"],c["beta2"]),
        use_grad_hooks=c.get("use_grad_hooks",False))
```

**Site 5 (train_lion, `:1450-1452`):**

OLD:
```python
        betas=(c["beta1"],0.99), weight_decay=c.get("lion_wd",3.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
```
NEW:
```python
        betas=(c["beta1"],0.99), weight_decay=c.get("lion_wd",3.0),
        use_grad_hooks=c.get("use_grad_hooks",False))
```

**Site 6 (train_prodigy, `:1510-1512`):**

OLD:
```python
    opt=Prodigy(m.parameters(), lr=c.get("prodigy_lr",1.0), weight_decay=c["weight_decay"],
        use_grad_hooks=c.get("use_grad_hooks",False))
    opt=_maybe_wrap_cuda_graph(opt, c)
```
NEW:
```python
    opt=Prodigy(m.parameters(), lr=c.get("prodigy_lr",1.0), weight_decay=c["weight_decay"],
        use_grad_hooks=c.get("use_grad_hooks",False))
```

> Net of Edit 2.9: −11 lines (4 def + 1 blank + 6 calls). `opt` value is identical at all
> 6 sites (the shim was pure identity), so this is transport-only / behavior-preserving.
> NOTE: `train_neuralgrok`, `train_supergrok`, `train_supergrok15`, `train_supergrok2`,
> and `train_looksam` never called the shim, so they are untouched (5 of the 11 loops).

---

# Verification / gate notes

* **mod-97 byte-identical (the hard gate):** `data_source` defaults `"modular"` →
  `make_data_for_task` runs the literal original 4 lines; `evaluate` `micro_batch=0` →
  original single-shot; `EarlyStopper` `mode="acc"` → original acc branch; `_eval_log`
  `early_stop_mode` absent → original `stop_acc`; `_fin` `_eval_micro_batch=0` →
  single-shot, `st.mode=="acc"` → original confirm branch. The dead-code removal is an
  `opt=opt` identity. Nothing on the mod-97 hot path changes value or order. No CLI flag
  is added, so `--gpus 2 --optimizers adamw ... --early-stop-max-steps 300` keeps the run
  fully `"modular"`.
* **A/A/A determinism:** the stub stream is seeded by `(seed, step, salt)` via a CPU
  `torch.Generator`; probes use fixed negative-step seeds and are materialized once →
  identical across the three replays. No `torch` global RNG is consumed by the stub
  (its own generators), so model-init RNG (`get_init_state` → `torch.manual_seed(seed)`)
  is unperturbed.
* **fp64 (decoder 1e-4) / grad rel / L3-TC megakernel:** untouched. The gate exercises the
  mod-97 path, which is the original code. `evaluate`'s sum/÷n micro-batch is only
  reachable when `0 < micro_batch < n` (scaled path), and is a transport-only restatement
  of the mean.
* **gfx942 / tpu:** no `csrc`/dispatch/kernel edit; preserved. NEW knobs are pure-Python,
  DEFAULT-OFF, and `#if`-style erasable (every NEW branch collapses to the pre-knob line
  when the knob is at its default).
* **Importable interface, no downloads:** `dataset_sources.py` imports only `torch`;
  stub bodies are synthetic. Real loaders drop in behind `make_source_for_task` /
  `_*_stub` with the identical signature. The module is imported LAZILY (inside the
  non-modular branch only), so `import grokking_race_v2` never imports it.
* **Dead-code KEEP proof:** `tests/hw/test_mamba_megakernel.py:37-39` does
  `import grokking_race_v2 as g; g.MambaModel(p=97, ntok=99, seq_len=8, d=128, nl=2, ...)`;
  `mamba_oracle.py` / `mamba3_oracle.py` transcribe `SelectiveSSMLayer`. KEEP both.
  `_maybe_checkpoint` is called in `Transformer.forward`/`ViT.forward`/`MambaModel.forward`.
  KEEP. ONLY `_maybe_wrap_cuda_graph` is removed.

## Self-check commands (lead, after apply)

```
python -c "import ast; ast.parse(open('grokking_race_v2.py').read())"
python -c "import ast; ast.parse(open('grokking_optimizers/dataset_sources.py').read())"
python -c "import grokking_race_v2"
python -c "import grokking_optimizers.dataset_sources as d; \
  tx,ty,vx,vy,ex,ey=d.make_source_for_task({'model_type':'decoder','data_source':'fineweb_edu','num_tokens':99,'eval_probe_rows':256,'train_batch_size':32,'early_stop_max_steps':128}, 1); \
  print(tx.shape, ty.shape, vx.shape, ex.shape)"
grep -n "_maybe_wrap_cuda_graph" grokking_race_v2.py   # → MUST be empty after apply
python -m pytest tests/hw/test_mamba_megakernel.py -k "oracle or mirror or layout" -q  # MambaModel still LIVE
python grokking_race_v2.py --gpus 2 --optimizers adamw --num-seeds 1 --early-stop-max-steps 300 --no-status-server --output /tmp/smoke
```

The smoke run uses no `--data-source` flag, so `data_source` stays `"modular"` and the
run is byte-identical to pre-change (the hard gate). The `dataset_sources` stub is only
import-exercised by the optional self-check command, never by the gate (it is imported
lazily inside the non-modular branch).
