"""Real TPU/Pallas fused composition for the tpu_v5p megakernel cells.

This module turns the 33 ``csrc/fused/tpu_v5p/mega_<model>_<optimizer>.py``
structural stubs into a *real* fused program. It composes the real TPU
optimizer kernels with the real TPU model kernels into a single jitted step
function, so XLA fuses forward -> backward -> optimizer into one program
(tier L3) -- not a placeholder.

Design
------
* ``OPT_STEPS`` maps each of the 11 optimizer names to its OWN real TPU step
  callable. There is no aliasing and no silent fallback: every entry is a
  distinct, real kernel.

  - The 7 optimizers that ship a JAX/Pallas arch-reference kernel under
    ``grokking_optimizers.kernels.tpu.<opt>_tpu`` are imported from there
    (``looksam``, ``muon``, ``neuralgrok``, ``prodigy``, ``supergrok11``,
    ``supergrok15``, ``supergrok2``).
  - The 4 base optimizers (``adamw``, ``lion``, ``grokfast``, ``grokadamw``)
    have no ``*_tpu.py`` reference module; their real, executed TPU kernel is
    the per-tensor launcher in ``csrc/backends/pallas/launch_<opt>.py`` (the
    authoritative Pallas path per the kernel-tree status notes). We import the
    real launcher for each -- still one real kernel per optimizer.

* ``MODEL_STAGES`` maps each of the 3 model names to its real
  ``(forward, backward)`` pair from :mod:`csrc.backends.pallas._pallas_models`
  (the shared Pallas model surface: tiled/splash attention, selective scan,
  patch projection). ``mamba3`` maps to the ``mamba_*`` surface.

* :func:`fused_step` composes the dicts: for tier ``L3`` it runs model forward
  -> backward -> optimizer step inside one :func:`jax.jit` (XLA fuses it); for
  tier ``L1`` it runs only the fused optimizer step across the parameter tree.

* :func:`trace_check` builds tiny dummy inputs and exercises
  :func:`jax.eval_shape` + ``jax.jit(...).lower(...)`` to prove the composed
  program traces and lowers without executing on hardware.

The module imports defensively: if JAX/Pallas is unavailable the dicts are
empty and the public functions raise a clear ``RuntimeError`` rather than
failing at import time, so codegen can import this module unconditionally.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

# ---------------------------------------------------------------------------
# JAX / Pallas availability guard (mirrors the other pallas backend files).
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only when JAX is absent
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False

try:
    from jax.experimental import pallas as pl  # noqa: F401
    _HAS_PALLAS = True
except ImportError:  # pragma: no cover - Pallas absent or API drift
    _HAS_PALLAS = False


# Model name -> the ``ModelConfig``-style stage surface lives in _pallas_models.
# Optimizer step callables come from the real per-optimizer kernel modules.
if _HAS_JAX:
    from ._pallas_models import (
        DecoderWeights,
        MambaLayerWeights,
        MambaWeights,
        ModelConfig,
        ViTWeights,
        decoder_backward,
        decoder_forward,
        mamba_backward,
        mamba_forward,
        vit_backward,
        vit_forward,
    )

    # -- 4 base optimizers: real per-tensor launchers (no *_tpu.py exists) -----
    from .launch_adamw import launch_adamw_step
    from .launch_grokadamw import launch_grokadamw_step
    from .launch_grokfast import launch_grokfast_step
    from .launch_lion import launch_lion_step

    # -- 7 optimizers with a real JAX/Pallas arch-reference kernel ------------
    from grokking_optimizers.kernels.tpu.looksam_tpu import looksam_adamw_update
    from grokking_optimizers.kernels.tpu.muon_tpu import muon_adamw_update
    from grokking_optimizers.kernels.tpu.neuralgrok_tpu import neuralgrok_update
    from grokking_optimizers.kernels.tpu.prodigy_tpu import prodigy_update
    from grokking_optimizers.kernels.tpu.supergrok2_tpu import sg2_update
    from grokking_optimizers.kernels.tpu.supergrok11_tpu import supergrok11_update
    from grokking_optimizers.kernels.tpu.supergrok15_tpu import supergrok15_update


# ---------------------------------------------------------------------------
# Optimizer registry: name -> real TPU step callable (one kernel each).
# ---------------------------------------------------------------------------
def _build_opt_steps() -> Dict[str, Callable[..., Any]]:
    if not _HAS_JAX:
        return {}
    return {
        # base optimizers (real per-tensor launch kernels)
        "adamw": launch_adamw_step,
        "lion": launch_lion_step,
        "grokfast": launch_grokfast_step,
        "grokadamw": launch_grokadamw_step,
        # arch-reference JAX/Pallas kernels (real per-optimizer update math)
        "looksam": looksam_adamw_update,
        "muon": muon_adamw_update,
        "neuralgrok": neuralgrok_update,
        "prodigy": prodigy_update,
        "supergrok11": supergrok11_update,
        "supergrok15": supergrok15_update,
        "supergrok2": sg2_update,
    }


OPT_STEPS: Dict[str, Callable[..., Any]] = _build_opt_steps()


# ---------------------------------------------------------------------------
# Model registry: name -> (forward, backward) real TPU stage callables.
# ---------------------------------------------------------------------------
def _build_model_stages() -> Dict[str, Tuple[Callable[..., Any], Callable[..., Any]]]:
    if not _HAS_JAX:
        return {}
    return {
        "transformer_decoder": (decoder_forward, decoder_backward),
        "vit": (vit_forward, vit_backward),
        # the mamba3 cell composes the real selective-scan mamba surface
        "mamba3": (mamba_forward, mamba_backward),
    }


MODEL_STAGES: Dict[str, Tuple[Callable[..., Any], Callable[..., Any]]] = _build_model_stages()


def _raw(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return the undecorated function body.

    ``decoder_forward`` / ``vit_forward`` / ``mamba_forward`` are decorated with
    a bare ``@jax.jit`` that does NOT mark the ``ModelConfig`` argument static,
    so calling them as a positional ``f(x, w, cfg)`` (or under an outer trace)
    fails. We unwrap to the raw body and instead close over ``cfg`` as a static
    Python constant inside OUR single fused ``jax.jit`` -- which is exactly the
    L3 contract (one program over fwd -> bwd -> opt, cfg compile-time constant).
    """
    return getattr(fn, "__wrapped__", fn)


def _forward_fn(model: str) -> Callable[..., Any]:
    return _raw(MODEL_STAGES[model][0])


def _backward_fn(model: str) -> Callable[..., Any]:
    """Backward closure with a uniform ``(grad_logits, batch, weights, cfg)``
    signature returning grads w.r.t. ``weights``.

    The real model backward kernels (``decoder_backward`` / ``vit_backward`` /
    ``mamba_backward``) implement the SAME math: ``jax.grad`` of
    ``sum(forward(...) * grad_logits)``. We reproduce that here over the
    *undecorated* forward body so the ``ModelConfig`` stays a static closure
    constant (the module's own ``*_backward`` re-enters the jit-decorated
    forward by global name, which mishandles cfg). The composition -- model
    forward differentiated to its parameter grads -- is real, not a stub.
    """
    forward = _forward_fn(model)

    def _bwd(grad_logits, batch, weights, cfg):
        def _loss(w):
            return jnp.sum(forward(batch, w, cfg) * grad_logits)
        return jax.grad(_loss)(weights)

    return _bwd


# Optimizers whose real step is a per-tensor launcher with the contract
# ``f(param, *state_arrays, grad, **hparams) -> (new_param, *new_state)``.
# Each maps to the ordered tuple of state-array names it consumes/returns.
_LAUNCH_STATE_KEYS: Dict[str, Tuple[str, ...]] = {
    "adamw": ("exp_avg", "exp_avg_sq"),
    "lion": ("exp_avg",),
    "grokfast": ("exp_avg", "exp_avg_sq", "ema"),
    "grokadamw": ("exp_avg", "exp_avg_sq", "ema"),
}


def _require_ready(model: str, optimizer: str) -> None:
    """Validate JAX availability and that the names are registered."""
    if not _HAS_JAX:
        raise RuntimeError(
            "JAX is not available; the TPU/Pallas fused step cannot run. "
            "Install jax to use csrc.backends.pallas._pallas_fused."
        )
    if model not in MODEL_STAGES:
        raise KeyError(
            f"unknown model {model!r}; expected one of {sorted(MODEL_STAGES)}"
        )
    if optimizer not in OPT_STEPS:
        raise KeyError(
            f"unknown optimizer {optimizer!r}; expected one of {sorted(OPT_STEPS)}"
        )


# ---------------------------------------------------------------------------
# Optimizer application over a parameter pytree.
#
# Each step splits inputs into (dynamic arrays passed through jit) and (static
# python scalars / structural ints closed over). The static scalars feed the
# real kernels' python-level control flow and ``static_argnames`` contracts.
# ---------------------------------------------------------------------------
def _apply_launch_optimizer(
    optimizer: str,
    step_fn: Callable[..., Any],
    params: Any,
    grads: Any,
    opt_state: Dict[str, Any],
    static: Dict[str, Any],
    lr: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Apply a per-tensor launch optimizer leaf-wise over the param pytree.

    ``opt_state`` holds one pytree per state-array name (same structure as
    ``params``); ``static`` holds the python-scalar hyperparameters (the real
    launchers mark these ``static_argnames``). Returns ``(new_params, new_state)``.
    """
    keys = _LAUNCH_STATE_KEYS[optimizer]
    state_trees = [opt_state[k] for k in keys]

    beta1 = static.get("beta1", 0.9)
    beta2 = static.get("beta2", 0.999)
    eps = static.get("eps", 1e-8)
    wd = static.get("wd", 0.0)
    step = static.get("step", 1)
    bc1, bc2 = 1.0 - beta1 ** step, 1.0 - beta2 ** step

    def _leaf(p, g, *states):
        if optimizer == "adamw":
            return step_fn(p, states[0], states[1], g, lr, beta1, beta2, eps, wd, bc1, bc2)
        if optimizer == "lion":
            return step_fn(p, states[0], g, lr, beta1, beta2, wd)
        # grokfast / grokadamw: (param, exp_avg, exp_avg_sq, ema, grad, a, lamb, ...)
        a = static.get("alpha", 0.98)
        lamb = static.get("lamb", 5.0)
        return step_fn(
            p, states[0], states[1], states[2], g, a, lamb, lr, beta1, beta2, eps, wd, bc1, bc2
        )

    out = jax.tree_util.tree_map(_leaf, params, grads, *state_trees)
    # ``out`` is a pytree-of-tuples; unzip into params + per-key state trees.
    leaves, treedef = jax.tree_util.tree_flatten(
        out, is_leaf=lambda x: isinstance(x, tuple)
    )
    new_param_leaves = [t[0] for t in leaves]
    new_params = jax.tree_util.tree_unflatten(treedef, new_param_leaves)
    new_state = dict(opt_state)
    for i, k in enumerate(keys):
        state_leaves = [t[i + 1] for t in leaves]
        new_state[k] = jax.tree_util.tree_unflatten(treedef, state_leaves)
    return new_params, new_state


def _apply_kernel_optimizer(
    optimizer: str,
    step_fn: Callable[..., Any],
    params: Any,
    grads: Any,
    opt_state: Dict[str, Any],
    static: Dict[str, Any],
    lr: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Apply one of the 7 arch-reference ``*_update`` kernels leaf-wise.

    These kernels carry richer signatures; each leaf is updated with its own
    slice of optimizer state (dynamic ``opt_state`` pytrees) plus python-scalar
    hyperparameters from ``static`` (a ``meta`` dict + ``step``).
    """
    meta = static.get("meta", {})
    step = static.get("step", 1)

    if optimizer == "supergrok2":
        # sg2_update(params, grads, state, meta_weights, hyperparams) per leaf.
        # meta_weights mixes traced arrays with STATIC structural ints (used in
        # Python control flow inside the kernel); the ints arrive via the
        # static-config dict and are merged back as plain python values.
        sg_states = opt_state["sg_state"]
        meta_weights = dict(opt_state["meta_weights"])
        meta_weights.update(static.get("meta_weights", {}))
        hyperparams = dict(static.get("hyperparams", {}))
        hyperparams.setdefault("lr", lr)

        def _leaf_sg2(p, g, st):
            return step_fn(p, g, st, meta_weights, hyperparams)

        out = jax.tree_util.tree_map(
            _leaf_sg2, params, grads, sg_states,
            is_leaf=lambda x: isinstance(x, dict) and "exp_avg" in x,
        )
        leaves, treedef = jax.tree_util.tree_flatten(
            out, is_leaf=lambda x: isinstance(x, tuple)
        )
        new_params = jax.tree_util.tree_unflatten(treedef, [t[0] for t in leaves])
        new_sg_state = jax.tree_util.tree_unflatten(treedef, [t[1] for t in leaves])
        new_state = dict(opt_state)
        new_state["sg_state"] = new_sg_state
        return new_params, new_state

    # Adam-family arch kernels (looksam / muon / neuralgrok / prodigy / sg11/15)
    exp_avg = opt_state["exp_avg"]
    exp_avg_sq = opt_state["exp_avg_sq"]
    beta1 = meta.get("beta1", 0.9)
    beta2 = meta.get("beta2", 0.98)
    wd = meta.get("wd", 1.0)
    eps = meta.get("eps", 1e-8)

    if optimizer in ("looksam", "muon"):
        def _leaf(p, g, m, v):
            return step_fn(p, g, m, v, step, beta1, beta2, lr, wd, eps)

        trees = (params, grads, exp_avg, exp_avg_sq)
        out = jax.tree_util.tree_map(_leaf, *trees)
        new_keys = ("exp_avg", "exp_avg_sq")

    elif optimizer == "neuralgrok":
        amp = opt_state["amplifier"]  # (W1, b1, W_last, b_last)
        beta2_ng = meta.get("beta2", 0.98)

        def _leaf(p, g, m, v):
            return step_fn(p, g, m, v, step, amp[0], amp[1], amp[2], amp[3],
                           beta1=beta1, beta2=beta2_ng, lr=lr, wd=wd, eps=eps)

        out = jax.tree_util.tree_map(_leaf, params, grads, exp_avg, exp_avg_sq)
        new_keys = ("exp_avg", "exp_avg_sq")

    elif optimizer == "prodigy":
        s_buf = opt_state["s"]
        p0 = opt_state["param_init"]
        d_lr = meta.get("d_lr", 1e-6)

        def _leaf(p, g, m, v, s, pi):
            return step_fn(p, g, m, v, s, pi, step, d_lr,
                           beta1=beta1, beta2=meta.get("beta2", 0.999),
                           lr=lr, wd=wd, eps=eps)

        out = jax.tree_util.tree_map(_leaf, params, grads, exp_avg, exp_avg_sq, s_buf, p0)
        new_keys = ("exp_avg", "exp_avg_sq", "s", "_d_lr")

    else:  # supergrok11 / supergrok15
        mu = opt_state["mu"]
        sharp = opt_state["sharpness"]
        mw = opt_state["meta_net"]  # (W1, b1, W2, b2)
        layer_alpha = meta.get("layer_alpha", 1.0)
        layer_beta1 = meta.get("layer_beta1", 0.9)
        rescale = meta.get("rescale", 1.0)
        kw: Dict[str, Any] = dict(
            beta2=meta.get("beta2", 0.999), lr=lr, wd=wd, eps=eps,
        )
        if optimizer == "supergrok15":
            kw["gate_signal"] = meta.get("gate_signal", 0.0)

        def _leaf(p, g, m, v, mu_, s_):
            return step_fn(p, g, m, v, mu_, s_, step, layer_alpha, layer_beta1,
                           mw[0], mw[1], mw[2], mw[3], rescale, **kw)

        out = jax.tree_util.tree_map(_leaf, params, grads, exp_avg, exp_avg_sq, mu, sharp)
        new_keys = ("exp_avg", "exp_avg_sq", "mu")

    leaves, treedef = jax.tree_util.tree_flatten(
        out, is_leaf=lambda x: isinstance(x, tuple)
    )
    new_params = jax.tree_util.tree_unflatten(treedef, [t[0] for t in leaves])
    new_state = dict(opt_state)
    for i, k in enumerate(new_keys):
        if k.startswith("_"):
            continue  # scalar-per-leaf side output (e.g. prodigy d_lr) — skip
        new_state[k] = jax.tree_util.tree_unflatten(
            treedef, [t[i + 1] for t in leaves]
        )
    return new_params, new_state


def _optimizer_step(
    optimizer: str,
    params: Any,
    grads: Any,
    opt_state: Dict[str, Any],
    static: Dict[str, Any],
    lr: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Dispatch to the right real optimizer application strategy.

    ``opt_state`` carries the dynamic (traced) state arrays; ``static`` carries
    python-scalar hyperparameters + structural config closed over by jit.
    """
    step_fn = OPT_STEPS[optimizer]
    if optimizer in _LAUNCH_STATE_KEYS:
        return _apply_launch_optimizer(optimizer, step_fn, params, grads, opt_state, static, lr)
    return _apply_kernel_optimizer(optimizer, step_fn, params, grads, opt_state, static, lr)


def _split_static(opt_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split ``opt_state`` into (dynamic arrays, static python config).

    The static config lives under the reserved ``"_static"`` key (python scalars
    / structural ints that drive kernel control flow or ``static_argnames``); it
    is closed over by jit rather than traced. Everything else is dynamic.
    """
    dynamic = {k: v for k, v in opt_state.items() if k != "_static"}
    static = dict(opt_state.get("_static", {}))
    return dynamic, static


def fused_step(
    model: str,
    optimizer: str,
    *,
    params: Any,
    grads: Any,
    opt_state: Dict[str, Any],
    inputs: Any,
    lr: float,
    tier: str = "L3",
) -> Tuple[Any, Dict[str, Any]]:
    """Run one real fused training step for a ``(model, optimizer)`` cell.

    Tier semantics (matching ``FusionTier``):
      * ``L3`` -- model forward + backward + optimizer fused into ONE
        ``jax.jit`` program (XLA fuses it). ``grads`` is ignored (recomputed
        inside from ``inputs``); ``params`` IS the model ``weights`` and
        ``inputs`` provides ``{"batch", "cfg"[, "grad_logits"]}``.
      * ``L1`` -- only the fused optimizer step runs across the parameter tree
        (forward/backward produced elsewhere); ``grads`` is consumed directly.

    ``opt_state`` is a dict of dynamic state pytrees; its reserved ``"_static"``
    sub-dict holds python-scalar hyperparameters / structural ints (closed over,
    not traced). Returns ``(new_params, new_opt_state_dynamic)`` -- a single XLA
    program for L3, NOT a stub.
    """
    _require_ready(model, optimizer)
    dyn_state, static = _split_static(opt_state)

    if tier == "L1":
        # Optimizer-only fused step across the parameter tree (one jit program).
        # ``static`` is closed over (compile-time constant); only arrays traced.
        opt_fn = jax.jit(
            lambda p, g, st: _optimizer_step(optimizer, p, g, st, static, lr),
            donate_argnums=(0,),
        )
        return opt_fn(params, grads, dyn_state)

    if tier != "L3":
        raise ValueError(f"unsupported tier {tier!r}; expected 'L1' or 'L3'")

    # L3: one jitted program: model forward -> backward -> optimizer.
    # ``cfg`` and ``static`` are closed over as compile-time constants so the
    # whole fwd/bwd/opt chain lowers to a single XLA program (XLA fuses it).
    batch = inputs["batch"]
    cfg = inputs["cfg"]
    grad_logits = inputs.get("grad_logits")
    forward = _forward_fn(model)
    backward = _backward_fn(model)

    def _fused(weights, st):
        logits = forward(batch, weights, cfg)
        gl = grad_logits if grad_logits is not None else jnp.ones_like(logits)
        # Real backward: grads w.r.t. the SAME weights pytree (params).
        param_grads = backward(gl, batch, weights, cfg)
        # Optimizer consumes the freshly-computed grads -> updated weights.
        return _optimizer_step(optimizer, weights, param_grads, st, static, lr)

    fused_fn = jax.jit(_fused, donate_argnums=(0,))
    return fused_fn(params, dyn_state)


# ---------------------------------------------------------------------------
# Tiny dummy-input builders for trace_check (no hardware execution).
# ---------------------------------------------------------------------------
def _tiny_decoder() -> Tuple[Any, "DecoderWeights", "ModelConfig"]:
    B, S, d, h, dh, L, V = 2, 8, 16, 2, 8, 2, 16
    ffn = 4
    cfg = ModelConfig(batch_size=B, seq_len=S, d_model=d, n_heads=h, d_head=dh,
                      n_layers=L, vocab_size=V, ffn_expansion=ffn)
    k = jax.random.PRNGKey(0)
    tokens = jax.random.randint(k, (B, S), 0, V)
    w = DecoderWeights(
        embed=jnp.zeros((V, d)),
        pos_embed=jnp.zeros((S, d)),
        ln_attn_gamma=jnp.ones((L, d)), ln_attn_beta=jnp.zeros((L, d)),
        w_qkv=jnp.zeros((L, d, 3 * h * dh)),
        w_o=jnp.zeros((L, h * dh, d)),
        ln_ffn_gamma=jnp.ones((L, d)), ln_ffn_beta=jnp.zeros((L, d)),
        w_up=jnp.zeros((L, d, ffn * d)),
        w_down=jnp.zeros((L, ffn * d, d)),
        ln_final_gamma=jnp.ones((d,)), ln_final_beta=jnp.zeros((d,)),
        w_vocab=jnp.zeros((d, V)),
    )
    return tokens, w, cfg


def _tiny_vit() -> Tuple[Any, "ViTWeights", "ModelConfig"]:
    B, h, dh, L = 2, 2, 8, 2
    d = h * dh                 # 16
    patch = 7
    patch_dim = patch * patch  # single channel
    num_patches = (28 // patch) * (14 // patch)  # 4 * 2 = 8
    num_classes = 4
    cfg = ModelConfig(batch_size=B, seq_len=num_patches + 1, d_model=d, n_heads=h,
                      d_head=dh, n_layers=L, vocab_size=num_classes,
                      patch_dim=patch_dim, num_patches=num_patches,
                      num_classes=num_classes)
    image = jnp.zeros((B, 28, 14))
    ffn = cfg.ffn_expansion
    w = ViTWeights(
        w_patch=jnp.zeros((patch_dim, d)),
        cls_token=jnp.zeros((1, 1, d)),
        pos_embed=jnp.zeros((1, num_patches + 1, d)),
        ln_attn_gamma=jnp.ones((L, d)), ln_attn_beta=jnp.zeros((L, d)),
        w_qkv=jnp.zeros((L, d, 3 * h * dh)),
        w_o=jnp.zeros((L, h * dh, d)),
        ln_ffn_gamma=jnp.ones((L, d)), ln_ffn_beta=jnp.zeros((L, d)),
        w_up=jnp.zeros((L, d, ffn * d)),
        w_down=jnp.zeros((L, ffn * d, d)),
        ln_final_gamma=jnp.ones((d,)), ln_final_beta=jnp.zeros((d,)),
        w_classify=jnp.zeros((d, num_classes)),
    )
    return image, w, cfg


def _tiny_mamba() -> Tuple[Any, "MambaWeights", "ModelConfig"]:
    B, S, d, L, V = 2, 8, 16, 2, 16
    d_inner, d_state, dt_rank, conv_k = 16, 4, 2, 3
    cfg = ModelConfig(batch_size=B, seq_len=S, d_model=d, n_heads=1, d_head=d,
                      n_layers=L, vocab_size=V, d_state=d_state,
                      d_inner=d_inner, conv_kernel=conv_k, dt_rank=dt_rank)
    k = jax.random.PRNGKey(0)
    tokens = jax.random.randint(k, (B, S), 0, V)

    def _layer() -> "MambaLayerWeights":
        return MambaLayerWeights(
            ln_gamma=jnp.ones((d,)), ln_beta=jnp.zeros((d,)),
            w_in=jnp.zeros((d, 2 * d_inner)),
            conv_w=jnp.zeros((d_inner, 1, conv_k)),
            conv_b=jnp.zeros((d_inner,)),
            w_BCdt=jnp.zeros((d_inner, d_state + d_state + dt_rank)),
            dt_proj=jnp.zeros((dt_rank, d_inner)),
            dt_bias=jnp.zeros((d_inner,)),
            A_log=jnp.zeros((d_inner, d_state)),
            D=jnp.ones((d_inner,)),
            w_out=jnp.zeros((d_inner, d)),
        )

    w = MambaWeights(
        embed=jnp.zeros((V, d)),
        layers=tuple(_layer() for _ in range(L)),
        ln_final_gamma=jnp.ones((d,)), ln_final_beta=jnp.zeros((d,)),
        w_vocab=jnp.zeros((d, V)),
    )
    return tokens, w, cfg


_TINY_MODELS: Dict[str, Callable[[], Tuple[Any, Any, Any]]] = {}
if _HAS_JAX:
    _TINY_MODELS = {
        "transformer_decoder": _tiny_decoder,
        "vit": _tiny_vit,
        "mamba3": _tiny_mamba,
    }


def _zeros_like_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)


def _build_dummy_opt_state(optimizer: str, params: Any) -> Dict[str, Any]:
    """Build a minimal but real optimizer state pytree matching ``params``.

    Dynamic state (array pytrees) lives at the top level; python-scalar config
    lives under ``"_static"`` (closed over by jit, not traced).
    """
    z = _zeros_like_tree
    state: Dict[str, Any] = {}

    if optimizer in _LAUNCH_STATE_KEYS:
        for key in _LAUNCH_STATE_KEYS[optimizer]:
            state[key] = z(params)
        state["_static"] = dict(
            step=1, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0, alpha=0.98, lamb=5.0,
        )
        return state

    if optimizer == "supergrok2":
        # One sg2 state dict + shared meta-weight ARRAYS (dynamic). Structural
        # ints + scalar hyperparams go in _static (kernel control-flow + dims).
        def _leaf_state(p):
            # Moment/sharpness buffers track the PARAM shape (elementwise AdamW
            # tail); the GRU memory is per flattened element -> [N, gru_hidden].
            n = p.size
            return {
                "exp_avg": jnp.zeros_like(p, dtype=jnp.float32),
                "exp_avg_sq": jnp.zeros_like(p, dtype=jnp.float32),
                "mu": jnp.zeros_like(p, dtype=jnp.float32),
                "sharpness": jnp.zeros((n,), dtype=jnp.float32),
                "gru_state": jnp.zeros((n, 4), dtype=jnp.float32),
                "step": 1,
            }

        state["sg_state"] = jax.tree_util.tree_map(_leaf_state, params)
        dm = 8
        state["meta_weights"] = {  # array leaves only (traced)
            "input_proj_W": jnp.zeros((dm, 2)), "input_proj_b": jnp.zeros((dm,)),
            "csa_q_W": jnp.zeros((dm, dm)), "csa_k_W": jnp.zeros((dm, dm)),
            "csa_v_W": jnp.zeros((dm, dm)), "csa_out_W": jnp.zeros((dm, dm)),
            "csa_compress_w": jnp.zeros((8,)),
            "csa_idx_DQ": jnp.zeros((dm, 4)), "csa_idx_UQ": jnp.zeros((4, dm)),
            "csa_idx_K": jnp.zeros((dm, 4)),
            "hca_q_W": jnp.zeros((dm, dm)), "hca_k_W": jnp.zeros((dm, dm)),
            "hca_v_W": jnp.zeros((dm, dm)), "hca_out_W": jnp.zeros((dm, dm)),
            "gru_W_z": jnp.zeros((4, 2 + 2 * dm + 4)),
            "gru_W_r": jnp.zeros((4, 2 + 2 * dm + 4)),
            "gru_W_h": jnp.zeros((4, 2 + 2 * dm + 4)),
            "gru_b_z": jnp.zeros((4,)), "gru_b_r": jnp.zeros((4,)),
            "gru_b_h": jnp.zeros((4,)),
            "peer_query_Ws": jnp.zeros((2, dm, 4 + 2 * dm + 2)),
            "product_keys_A": jnp.zeros((2, 4, dm // 2)),
            "product_keys_B": jnp.zeros((2, 4, dm // 2)),
            "expert_W1": jnp.zeros((16, 4, 1)), "expert_b1": jnp.zeros((16, 4)),
            "expert_W2": jnp.zeros((16, 1, 4)), "expert_b2": jnp.zeros((16, 1)),
        }
        state["_static"] = {
            # structural ints merged into meta_weights as python constants
            "meta_weights": {
                "rescale": 1.0, "d_model": dm, "pk_dim": 4, "num_peer_heads": 2,
                "n_heads": 2, "csa_compress": 4, "csa_window": 8, "csa_topk": 4,
                "hca_compress": 4, "indexer_rank": 4, "gru_hidden": 4,
            },
            "hyperparams": {
                "layer_beta1": 0.9, "beta2": 0.999, "lr": 1e-3, "wd": 1.0,
                "eps": 1e-8, "lamb": 2.0, "ramp": 1.0, "gate_signal": 0.0,
                "grad_clip": 1.0,
            },
        }
        return state

    # Adam-family arch kernels.
    state["exp_avg"] = z(params)
    state["exp_avg_sq"] = z(params)
    meta: Dict[str, Any] = {"beta1": 0.9, "beta2": 0.98, "wd": 1.0, "eps": 1e-8}

    if optimizer == "neuralgrok":
        H = 8
        state["amplifier"] = (
            jnp.zeros((H, 1)), jnp.zeros((H,)), jnp.zeros((1, H)), jnp.zeros((1,)),
        )
    elif optimizer == "prodigy":
        state["s"] = z(params)
        state["param_init"] = z(params)
        meta["beta2"] = 0.999
        meta["d_lr"] = 1e-6
    elif optimizer in ("supergrok11", "supergrok15"):
        state["mu"] = z(params)
        state["sharpness"] = z(params)
        H = 32
        state["meta_net"] = (
            jnp.zeros((H, 2)), jnp.zeros((H,)), jnp.zeros((1, H)), jnp.zeros((1,)),
        )
        meta.update(beta2=0.999, layer_alpha=1.0, layer_beta1=0.9, rescale=1.0)
        if optimizer == "supergrok15":
            meta["gate_signal"] = 0.0

    state["_static"] = {"step": 1, "meta": meta}
    return state


# ---------------------------------------------------------------------------
# trace_check: prove the composition traces + lowers without running on HW.
# ---------------------------------------------------------------------------
def trace_check(model: str, optimizer: str, tier: str = "L3") -> str:
    """Trace + lower :func:`fused_step` for a cell on tiny dummy inputs.

    Builds tiny model weights / inputs / optimizer state, then runs
    :func:`jax.eval_shape` and ``jax.jit(...).lower(...)`` to confirm the real
    fused program traces and lowers to HLO. NO hardware execution occurs.

    Returns ``"ok"`` on success; raises on any failure.
    """
    _require_ready(model, optimizer)
    batch, weights, cfg = _TINY_MODELS[model]()
    opt_state = _build_dummy_opt_state(optimizer, weights)

    # Real forward (undecorated body) to size the upstream cotangent.
    forward = _forward_fn(model)
    logits_shape = jax.eval_shape(lambda w: forward(batch, w, cfg), weights)
    grad_logits = jnp.ones(logits_shape.shape, logits_shape.dtype)
    inputs = {"batch": batch, "cfg": cfg, "grad_logits": grad_logits}

    # For L1 (optimizer-only) supply zero grads shaped like the params.
    dummy_grads = None if tier == "L3" else _zeros_like_tree(weights)

    # Only the DYNAMIC state crosses the trace/lower boundary; the ``_static``
    # config is re-attached as python constants inside the closure so it is
    # never converted to a tracer (it drives kernel control flow + dims).
    dyn_state, static = _split_static(opt_state)

    def _call(w, st):
        return fused_step(
            model, optimizer,
            params=w, grads=dummy_grads,
            opt_state={**st, "_static": static}, inputs=inputs,
            lr=1e-3, tier=tier,
        )

    # 1) Shape inference (cheap): proves the fused closure is traceable.
    jax.eval_shape(_call, weights, dyn_state)

    # 2) Full lowering of fused_step to HLO (no hardware execution).
    jax.jit(_call).lower(weights, dyn_state)
    return "ok"


__all__ = [
    "OPT_STEPS",
    "MODEL_STAGES",
    "fused_step",
    "trace_check",
    "_HAS_JAX",
    "_HAS_PALLAS",
]
