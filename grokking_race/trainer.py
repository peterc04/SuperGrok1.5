"""One training run: a DeepSeek-V4.1-Flash model trained on one task under one optimizer.

Every optimizer gets the same model initialisation (seeded construction), the
same data split and the same step/eval schedule. The loop is full-batch
gradient descent, as in the previous races.

Grokking
    A run has grokked at step ``s`` if **test** accuracy is >= ``threshold``
    (0.95) at the eval at step ``s`` and at the next ``patience - 1`` evals
    (default 50 evals x 10 steps: held for 500 steps). A single spike does not
    count. There are two splits, train and test (``tasks.py``); test is only
    ever evaluated, never trained on. Train loss and accuracy are logged on
    the examples the optimizer takes gradient steps on (its meta slice, if it
    has one, excluded).

Cost to grok, three ways
    steps (optimizer updates), wall-clock seconds spent in training iterations
    (evaluation excluded; it is identical for every optimizer), and training
    FLOPs. FLOPs are counted by ``torch.utils.flop_counter.FlopCounterMode``
    (matrix-multiply FLOPs, the convention behind MFU) over *everything* an
    iteration runs: extra sharpness-aware or meta forward/backward passes,
    Newton-Schulz iterations, amplifier networks, and Shampoo's
    eigendecompositions (9 n^3 each, a formula the counter lacks). Each distinct kind of iteration
    an optimizer performs (``opt.upcoming_step_kind()``, e.g. a NeuralGrok meta
    step vs a plain step) is measured exactly the first time it occurs; later
    iterations of that kind reuse the count. An optimizer's own work can depend
    on which parameters have a gradient (an idle MoE expert gets none: Muon
    then skips its Newton-Schulz), so a kind is counted again whenever that set
    is new for it, until two sets have given the same count (the kind does not
    depend on it) or 16 counts are reached; iterations charged a count made
    with a different set are reported in ``flops_approx_steps``. The
    counter's own overhead is kept out of the timing: a measured iteration is
    charged the median time of the unmeasured iterations of the same kind, or,
    for a kind that never recurs, the best-timed kind's median scaled by the
    FLOP ratio.

Optimizer hooks (class attributes)
    ``needs_closure``: ``step()`` gets ``closure``, which zeroes the grads,
    recomputes the training loss at the *current* parameters, backpropagates and
    returns the loss (sharpness-aware methods, NeuralGrok's fresh gradient).
    ``needs_meta_loss``: ``step()`` gets ``meta_loss(params) -> Tensor``, the
    differentiable loss on a held-out meta batch with ``params`` (aligned with
    the optimizer's parameters, group by group) substituted into the model. The
    meta batch is carved out of the training split (``meta_frac``); the
    optimizer trains on the rest, and test data never drives training. With
    ``same_train_data`` every optimizer trains on that same rest, so the data is
    matched exactly (off by default: then optimizers without a held-out signal
    train on the whole train split). ``needs_train_meta_loss``: also
    ``train_meta_loss(params)``, the training cross-entropy at substituted
    parameters. ``uses_step_loss``: ``step()`` gets the iteration's ``loss``.
    ``wants_losses``: after every evaluation the loop calls
    ``opt.set_losses(train_loss, meta_batch_loss, train_acc=train_acc)``, with
    the training loss and accuracy on the part of the training split the
    optimizer actually trains on (the meta slice excluded).
    Every forward an optimizer runs itself leaves the MoE router's load
    statistics untouched.

After each optimizer step the loop calls ``model.update_router_bias()``, the
auxiliary-loss-free MoE load-balancing update, identically for every optimizer.
"""

from __future__ import annotations

import math
import statistics
import time
import zlib
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from .results import EarlyStopper, RunResult
from .tasks import Task, carve_meta_split


@dataclass
class RunConfig:
    preset: str = "tiny"
    device: str = "cpu"
    dtype: str = "fp32"  # fp32 | bf16 (autocast for training and evaluation; master weights stay fp32)
    compile: bool = False
    max_steps: int = 20_000
    eval_every: int = 10
    threshold: float = 0.95
    patience: int = 50
    eval_batch: int = 0  # >0: evaluate in chunks of this many examples
    micro_batches: int = 1  # split the full batch into this many chunks and accumulate gradients (memory only)
    meta_frac: float = 0.10
    flop_count_every_step: bool = False  # count FLOPs on every iteration (slow; exact ground truth for tests)
    same_train_data: bool = False  # True: every optimizer trains on the same inner split (meta slice held out for all)
    param_policy: str = "uniform"  # uniform (decay everything) | deepseek (DeepSeek's per-role decay / lr)
    model_overrides: dict = field(default_factory=dict)  # deepseek_v41.Config fields, e.g. engram_vocab_size
    peak_flops: float | None = None  # device peak FLOP/s for MFU (e.g. 989e12 for H100 SXM bf16 dense)
    hparams: dict = field(default_factory=dict)  # per-optimizer overrides
    progress: bool = True


def _eigh_flops(a_shape, *args, out_shape=None, **kwargs) -> int:
    """Symmetric eigendecomposition with eigenvectors, about 9 n^3 FLOPs (Golub & Van Loan, *Matrix
    Computations*, 4th ed., sec. 8.3). PyTorch's counter has no formula for it and would count 0."""
    *batch, n, _ = a_shape
    return 9 * n**3 * math.prod(batch)


MAX_FLOP_COUNTS_PER_KIND = 16  # FLOP-counted iterations per step kind, at most

# FLOP formulas the counter lacks, for work some optimizers do (Shampoo's inverse roots)
EXTRA_FLOP_FORMULAS = {torch.ops.aten._linalg_eigh: _eigh_flops}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(cfg: RunConfig, device):
    if cfg.dtype == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    if cfg.dtype != "fp32":
        raise ValueError(f"dtype must be fp32 or bf16, got {cfg.dtype!r}")
    return nullcontext()


@torch.no_grad()
def evaluate(model, x, y, num_classes: int, chunk: int = 0, amp=None) -> tuple[float, float]:
    """(mean cross-entropy, accuracy of argmax over the first ``num_classes`` logits), under ``amp`` if given."""
    n = len(y)
    chunk = n if chunk <= 0 else chunk
    loss_sum, correct = 0.0, 0
    for i in range(0, n, chunk):
        with amp if amp is not None else nullcontext():
            logits = model(x[i : i + chunk]).float()
        loss_sum += F.cross_entropy(logits, y[i : i + chunk], reduction="sum").item()
        correct += (logits[:, :num_classes].argmax(-1) == y[i : i + chunk]).sum().item()
    return loss_sum / n, correct / n


def run_seed(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def train_one(opt_name: str, seed: int, task: Task, cfg: RunConfig) -> RunResult:
    from deepseek_v41 import build_model
    from grokking_optimizers import build_optimizer, get_spec, state_bytes

    device = torch.device(cfg.device)
    spec = get_spec(opt_name)
    hp = {**spec.race_defaults, **cfg.hparams.get(opt_name, {})}
    r = RunResult(
        optimizer=opt_name,
        seed=seed,
        task=task.name,
        frac_train=0.0,
        preset=cfg.preset,
        dtype=cfg.dtype,
        hparams=hp,
    )

    # Same init for every optimizer: construction is seeded by the run seed only.
    torch.manual_seed(seed)
    model = build_model(cfg.preset, vocab_size=task.vocab_size, max_seq_len=task.seq_len, **cfg.model_overrides).to(
        device
    )
    r.model_params = model.num_params()
    r.active_params = model.num_active_params()
    fwd = torch.compile(model) if cfg.compile else model

    # Optimizer-internal randomness (meta-net init etc.) depends on (seed, optimizer), not on run order.
    torch.manual_seed(run_seed(f"{seed}:{opt_name}"))
    opt = build_optimizer(opt_name, model, policy=cfg.param_policy, **hp)

    t = task.to(device)
    x_tr, y_tr = t.x_train, t.y_train
    x_meta = y_meta = None
    if getattr(opt, "needs_meta_loss", False) or getattr(opt, "wants_losses", False) or cfg.same_train_data:
        x_tr, y_tr, x_meta, y_meta = carve_meta_split(t.x_train.cpu(), t.y_train.cpu(), cfg.meta_frac, seed)
        x_tr, y_tr, x_meta, y_meta = (a.to(device) for a in (x_tr, y_tr, x_meta, y_meta))
    r.train_examples, r.meta_examples = len(y_tr), 0 if y_meta is None else len(y_meta)

    param_list = [p for g in opt.param_groups for p in g["params"]]
    name_of = {p: n for n, p in model.named_parameters()}
    param_names = [name_of[p] for p in param_list]
    amp = _autocast(cfg, device)

    chunks = list(zip(x_tr.tensor_split(cfg.micro_batches), y_tr.tensor_split(cfg.micro_batches)))

    def closure():
        """Full-batch loss and gradient; with micro-batches, the same sum computed in chunks."""
        opt.zero_grad(set_to_none=True)
        total = 0.0
        with torch.enable_grad():
            for xb, yb in chunks:
                with amp:
                    logits, aux = fwd(xb, return_aux=True)  # aux: the MoE sequence-wise balance loss
                # both terms are means over the chunk's sequences: weight by chunk size for the full-batch mean
                loss = (F.cross_entropy(logits.float(), yb) + aux) * (len(yb) / len(y_tr))
                loss.backward()
                total = total + loss.detach()
        return total

    def inner_closure():  # extra forwards an optimizer runs must not feed the router's load statistics
        with model.frozen_router_stats():
            return closure()

    def meta_loss(params):
        with model.frozen_router_stats(), amp:
            logits = torch.func.functional_call(model, dict(zip(param_names, params)), (x_meta,))
        return F.cross_entropy(logits.float(), y_meta)

    def train_meta_loss(params):
        subst = dict(zip(param_names, params))
        total = 0.0
        with model.frozen_router_stats():
            for xb, yb in chunks:
                with amp:
                    logits = torch.func.functional_call(model, subst, (xb,))
                total = total + F.cross_entropy(logits.float(), yb) * (len(yb) / len(y_tr))
        return total

    def iteration():
        loss = closure()
        kwargs = {}
        if getattr(opt, "needs_closure", False):
            kwargs["closure"] = inner_closure
        if getattr(opt, "needs_meta_loss", False):
            kwargs["meta_loss"] = meta_loss
        if getattr(opt, "needs_train_meta_loss", False):
            kwargs["train_meta_loss"] = train_meta_loss
        if getattr(opt, "uses_step_loss", False):
            kwargs["loss"] = loss
        opt.step(**kwargs)
        model.update_router_bias()
        return loss

    def grad_signature() -> int:  # which parameters got a gradient (idle MoE experts get none)
        return sum(p.numel() for p in param_list if p.grad is not None)

    stopper = EarlyStopper(cfg.threshold, cfg.patience)
    flop_table: dict = {}  # (step kind, gradient signature) -> FLOPs of one such iteration
    flop_invariant: set = set()  # kinds counted under two gradient signatures with equal FLOPs
    n_measured: dict = {}  # step kind -> FLOP-counted iterations so far
    step_kinds, step_sigs, step_times = [], [], []  # per iteration; time None where the counter was running
    iter_flops: list = []  # per iteration, only with cfg.flop_count_every_step
    counter_times: dict = {}  # step kind -> wall time of its first counted iteration (includes counter overhead)
    last_sig = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    t_wall, step = time.time(), 0
    for step in range(1, cfg.max_steps + 1):
        evaluating = step % cfg.eval_every == 0 or step == 1 or step == cfg.max_steps
        if hasattr(opt, "track_stats"):  # per-step diagnostics only where they are read
            opt.track_stats = evaluating
        kind = opt.upcoming_step_kind() if hasattr(opt, "upcoming_step_kind") else "step"
        # count FLOPs for a kind's first iteration, and again whenever the set of parameters with a gradient
        # (idle MoE experts get none) is new for the kind, unless the kind has shown its FLOPs do not depend on it
        measure = (
            cfg.flop_count_every_step
            or kind not in n_measured
            or (
                kind not in flop_invariant
                and (kind, last_sig) not in flop_table
                and n_measured[kind] < MAX_FLOP_COUNTS_PER_KIND
            )
        )
        _sync(device)
        t0 = time.perf_counter()
        if measure:
            with FlopCounterMode(display=False, custom_mapping=EXTRA_FLOP_FORMULAS) as counter:
                loss = iteration()
        else:
            loss = iteration()
        _sync(device)
        sig = last_sig = grad_signature()
        step_kinds.append(kind)
        step_sigs.append(sig)
        step_times.append(None if measure else time.perf_counter() - t0)
        if measure:
            flops = counter.get_total_flops()
            iter_flops.append(flops)
            counter_times.setdefault(kind, time.perf_counter() - t0)
            n_measured[kind] = n_measured.get(kind, 0) + 1
            flop_table[(kind, sig)] = flops
            seen = {v for (k, _s), v in flop_table.items() if k == kind}
            if len(seen) == 1 and sum(k == kind for k, _s in flop_table) > 1:
                flop_invariant.add(kind)

        if not torch.isfinite(loss):
            r.stopping_reason = "non_finite_loss"
            break
        if evaluating:
            model.eval()
            # "train" = the examples this optimizer takes gradient steps on (its meta slice excluded)
            tl, ta = evaluate(fwd, x_tr, y_tr, task.num_classes, cfg.eval_batch, amp)
            el, ea = evaluate(fwd, t.x_test, t.y_test, task.num_classes, cfg.eval_batch, amp)
            if getattr(opt, "wants_losses", False):  # its training examples vs its held-out slice
                ml = evaluate(fwd, x_meta, y_meta, task.num_classes, cfg.eval_batch, amp)[0]
                opt.set_losses(tl, ml, train_acc=ta)
            model.train()
            r.steps.append(step)
            for k, v in zip(
                ("train_losses", "train_accs", "test_losses", "test_accs"),
                (tl, ta, el, ea),
            ):
                getattr(r, k).append(v)
            diag = getattr(opt, "diagnostics", None)
            if callable(diag):
                r.diagnostics.append({"step": step, **diag()})
            if cfg.progress and (step == 1 or step % (cfg.eval_every * 50) == 0):
                print(
                    f"    [{opt_name} s{seed}] step {step:>6}  train {ta:.3f}  test {ea:.3f}  loss {tl:.4f}",
                    flush=True,
                )
            if stopper.update(ea, step):
                r.stopping_reason = "test_acc_held"
                break
    else:
        r.stopping_reason = "max_steps"

    # ---- cost accounting: per-iteration time and FLOPs, cumulated ----
    by_kind: dict = {}
    for k, dt in zip(step_kinds, step_times):
        if dt is not None:
            by_kind.setdefault(k, []).append(dt)
    kind_flops = {}  # step kind -> FLOPs of its first counted iteration
    for (k, _s), v in flop_table.items():
        kind_flops.setdefault(k, v)
    if cfg.flop_count_every_step:
        charged = iter_flops
    else:
        charged = []
        for k, s in zip(step_kinds, step_sigs):
            f = flop_table.get((k, s))
            if f is None:  # a gradient set this kind was never counted with
                f = kind_flops[k]
                if k not in flop_invariant:
                    r.flops_approx_steps += 1
            charged.append(f)
    kind_time = {k: statistics.median(v) for k, v in by_kind.items()}
    ref = max(by_kind, key=lambda k: len(by_kind[k]), default=None)
    for k in kind_flops:  # a kind never timed without the counter: scale the best-timed kind by FLOPs
        if k not in kind_time:
            if ref is not None and kind_flops[ref] > 0:
                kind_time[k] = kind_time[ref] * kind_flops[k] / kind_flops[ref]
            else:
                kind_time[k] = counter_times[k]  # nothing else to go on (a run of one iteration per kind)
    times = [dt if dt is not None else kind_time[k] for k, dt in zip(step_kinds, step_times)]
    cum_time, cum_flops, acc_t, acc_f = [], [], 0.0, 0
    for dt, f in zip(times, charged):
        acc_t += dt
        acc_f += f
        cum_time.append(acc_t)
        cum_flops.append(acc_f)
    r.eval_train_time = [cum_time[s - 1] for s in r.steps]
    r.eval_flops = [cum_flops[s - 1] for s in r.steps]
    r.flops_per_step_kind = {str(k): v for k, v in kind_flops.items()}

    r.total_steps = step
    r.wall_time = time.time() - t_wall
    r.train_time = cum_time[-1] if cum_time else 0.0
    r.train_flops = cum_flops[-1] if cum_flops else 0
    r.ms_per_step = 1e3 * r.train_time / max(step, 1)
    r.tokens_per_sec = x_tr.numel() * step / max(r.train_time, 1e-9)
    if cfg.peak_flops:
        r.mfu = r.train_flops / max(r.train_time, 1e-9) / cfg.peak_flops
    if device.type == "cuda":
        r.peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    r.optimizer_state_bytes = state_bytes(opt)
    r.best_test_acc = stopper.best
    r.first_cross_step = stopper.first_cross_step
    r.grokked = stopper.grok_step is not None
    if r.steps:
        r.final_train_acc = r.train_accs[-1]
        r.final_test_loss, r.final_test_acc = r.test_losses[-1], r.test_accs[-1]
    if r.grokked:
        r.grok_step = stopper.grok_step
        r.grok_train_time = cum_time[r.grok_step - 1]
        r.grok_flops = cum_flops[r.grok_step - 1]
    return r
