# Toy Tune Project — a portability proof harness for the SuperGrok1.5 autotuner

A deliberately trivial, **non-grokking** CUDA project with exactly **one
tunable knob**, built to prove that the SuperGrok1.5 autotuner
(`grokking_optimizers/compile.py`) is portable: it should be able to tune a
completely different project's CUDA kernel with **zero edits to `compile.py`**,
driven only by a TOML config + a Python project hook.

The op is SAXPY: `y = a*x + b` over a 1-D float tensor, registered as a torch
custom op (`torch.ops.toy.saxpy`). The one knob is the CUDA launch block size,
exposed as the `-DTOY_TUNED_BLOCK=<N>` macro. The op is numerically identical
for every block size (it uses a grid-stride loop), so the knob changes only
performance/codegen, never the result — exactly what an autotuner needs.

## Files

| File | Purpose |
|------|---------|
| `toy_kernel.cu` | The CUDA extension. Single `TORCH_LIBRARY(toy, ...)` op + `__global__` SAXPY kernel. Block size is the tunable macro `TOY_TUNED_BLOCK` (default 256), used as the launch block dim + grid-stride. `static_assert` rejects values > 1024. |
| `tune_hook.py` | Implements the portable hook contract `run(*, so_path, model, optimizer, arch, regime, seed) -> {"output": np.ndarray, "elapsed_ms": float}`. Loads the `.so` via `torch.ops.load_library`, synthesizes a seeded/regime-scaled input, runs warmup + 50 CUDA-event-timed iters, returns the median ms + output. `model`/`optimizer` are unused (the contract is generic). |
| `compile_config.toml` | Project config: `[project] tune_hook`, `macro_prefix = "TOY_"`, `[sources] cuda_root`, restricted `[optimizers]`/`[models]`/`[archs]`. Keys match exactly what `compile.py`'s `load_config` / `apply_to_buildspec` read. |
| `toy_search_space.yaml` | The minimal `--search-space`: one dim `block` → macro `TOY_TUNED_BLOCK` over `[128, 256, 512, 1024]`. Validated by `compile.py`'s `load_yaml`. |

## The tune command

From the **repo root** (`/workspace/SuperGrok1.5`):

```bash
source /workspace/venv/bin/activate
source .regpressure/env.sh
export PYTHONPATH=$PWD            # so the hook 'examples.toy_tune_project.tune_hook:run' imports

cd examples/toy_tune_project
python -m grokking_optimizers.compile \
    --optimizer toy_saxpy \
    --model toy \
    --arch sm_90a \
    --config compile_config.toml \
    --search-space toy_search_space.yaml \
    --out /tmp/toy_build
```

`-O` / `-M` / `-A` are the short forms of `--optimizer` / `--model` / `--arch`.
`compile.py` runs the hook subprocess with `cwd = <repo root>`, which is why the
config uses the repo-root-relative dotted path `examples.toy_tune_project.
tune_hook:run` (robust regardless of the invocation directory).

### What success looks like (target state, after Phase 2)

1. `compile.py` resolves `toy_kernel.cu` as the only source (via the Tier-1
   zero-config auto-glob) and AOT-builds a strict-math reference `.so` with
   `-DTOY_TUNED_BLOCK=256`.
2. It captures the reference output by calling `tune_hook:run` on that `.so`
   (regime `normal`, seed 0) — already wired today via `_hook_capture`.
3. For each candidate block size in the search space it builds a variant `.so`
   with `-DTOY_TUNED_BLOCK=<N>`, **times it via the same `tune_hook`**, and
   **validates the hook's output against the reference** across the configured
   regimes (`normal`/`large`/`small`/`adversarial`).
4. The autotuner reports the fastest block size and writes the winner. All
   four variants validate `ok` (SAXPY is exact for every block size); they
   differ only in `elapsed_ms`.

## What is VERIFIED today (standalone, no Phase-2 fixes)

All of the following were run and pass on this H100 (sm_90a) host:

- **Standalone compile + correctness.** `toy_kernel.cu` compiles with `nvcc`
  against torch and is numerically correct for **every** block size
  (128/256/512/1024):
  ```bash
  python -c "import torch.utils.cpp_extension as e; \
    e.load(name='toy_saxpy', sources=['examples/toy_tune_project/toy_kernel.cu'], \
           extra_cuda_cflags=['-DTOY_TUNED_BLOCK=256'], is_python_module=False); \
    import torch; x=torch.randn(1024,device='cuda'); y=torch.empty_like(x); \
    torch.ops.toy.saxpy(y,x,2.0,1.0); print('ok', torch.allclose(y, 2*x+1))"
  # -> ok True
  ```
  NOTE: a pure-`TORCH_LIBRARY` op must be loaded with `is_python_module=False`
  (or `torch.ops.load_library` on the `.so`) — the bare `e.load(...)` default
  looks for a `PyInit_<name>` symbol that a `TORCH_LIBRARY`-only TU doesn't
  export. This is exactly how `compile.py` loads `torch_op` artifacts.

- **Hook contract.** `python tune_hook.py` self-check prints
  `elapsed_ms=... n=4194304 correct=True`.

- **Hook ↔ `_hook_capture` protocol.** Driving the hook through `compile.py`'s
  exact subprocess protocol (`"pkg.module:fn"` rsplit-on-`:`, import, call,
  save `.npy`, emit `HOOK_OK <json>`) returns a correct output and a valid
  `elapsed_ms`, including under the `large` regime with a non-zero seed.

- **Config + search-space schema.** `compile.py`'s own loaders accept both
  files:
  ```bash
  python -c "from grokking_optimizers import compile as C; from pathlib import Path; \
    cfg=C.load_config(Path('examples/toy_tune_project/compile_config.toml')); \
    sp=C.get_search_space(Path('examples/toy_tune_project/toy_search_space.yaml')); \
    print(cfg['project']['macro_prefix'], cfg['project']['tune_hook']); \
    print([C.resolve_macros({'block':v}, sp['sm_90a']['dims'], 'device') for v in (128,256,512,1024)])"
  ```
  → `macro_prefix=TOY_`, and the dim emits `-DTOY_TUNED_BLOCK=128/256/512/1024`.

## Portability gaps in `compile.py` that Phase 2 must close

These are documented here, **not fixed** (other agents own `compile.py`). Each
was confirmed against the current `compile.py` (line numbers approximate).

### Gap 1 — the tune-hook seam is only HALF-wired (the decisive blocker)

`compile.py` already has a portable tune-hook contract and uses it for **oracle
capture**: `BuildSpec.tune_hook` (~L7453), `_hook_capture` (~L12759), and the
`_resolve_ref` reference path (`if getattr(spec, "tune_hook", None): ...`,
~L13533). The contract matches this project's hook exactly.

But the **variant timing** path (`_make_variant_timer`, ~L13860–13894) and the
**variant numerical-validation** path (`_validate_against_regimes` →
`_dump_variant_output`, ~L12664) do **not** consult `spec.tune_hook`. They are
hardwired to grokking's optimizer-step shape:

- Timing uses `_time_variant_oneshot(variant_so, OPT_CLASS[spec.optimizer], ...)`
  / a persistent worker whose body does `from {python_package} import {opt_class}`,
  builds `OptCls([param], lr=1e-3)`, and calls `opt.step()` (`_TIMING_SCRIPT`,
  ~L14037). A SAXPY op has no such optimizer class.
- Validation uses `_render_arg_construction` to synthesize args for a discovered
  `torch_op` entry and `_dump_variant_output` — again the torch.op path, not the
  hook.

**Phase 2:** when `spec.tune_hook` is set, route variant timing AND variant
output-capture-for-validation through `_hook_capture` (it already returns both
`output` and `elapsed_ms`), bypassing `OPT_CLASS` / `opt.step()` /
`_render_arg_construction` entirely.

### Gap 2 — source discovery globs `csrc/fused/<arch>` unconditionally

`_resolve_sources` (~L9162) computes
`structured = bindings + launchers + models + fused`, where `fused` is globbed
from a **hardcoded** `REPO_ROOT/csrc/fused/<arch_subdir>/*.cu` (~L9237) that has
**no `[sources]` config key**. On `sm_90a` (arch subdir `sm_90`),
`csrc/fused/sm_90/*.cu` exists (grokking's L3 megakernel TUs), so `structured`
is non-empty even after this project re-points `cuda_root`/`bindings_dir`/
`algorithms_dir` at itself. Consequences:

- The Tier-1 zero-config auto-glob (`_auto_discover_sources`) — which *would*
  find `toy_kernel.cu` — never fires (it only runs when `structured` is empty).
- Grokking's four `csrc/fused/sm_90/*_tc_launcher.cu` / `sg2_meta_tail.cu` leak
  into the toy build.

VERIFIED: with this project's config, `_resolve_sources` returns exactly those
four `csrc/fused/sm_90/*.cu` files and **not** `toy_kernel.cu`.

**Phase 2:** make the fused-dir root config-driven (e.g. a `[sources] fused_root`
key), OR skip the `csrc/fused` glob when `source_roots["cuda"]` is overridden to
a non-default root, so a foreign project's single `.cu` is discoverable on any
arch.

### Gap 3 — `_validate` / `OPT_CLASS` use the hardcoded profile taxonomy, not the config

`build()` calls `_validate(spec)` (~L8440), which hard-checks
`spec.optimizer in OPTIMIZERS` and `spec.model in MODELS` against the **profile
module constants** — it ignores the config's `[optimizers].enabled` /
`[models].enabled`. Meanwhile the CLI `--optimizer`/`--model` `choices=` ARE
built from the config. So with this project's config:

- CLI accepts only `toy_saxpy` / `toy` (from config), but
- `_validate` rejects `toy_saxpy` because it isn't in the profile `OPTIMIZERS`.

VERIFIED at runtime:
`ValueError: optimizer='toy_saxpy' not in ['adamw', ..., 'supergrok2']`.
These two layers are mutually exclusive — **no** optimizer/model name can
satisfy both with a foreign config. The same hardcoded coupling appears wherever
`OPT_CLASS[spec.optimizer]` is indexed in the timing/validation path
(~L13134/13893/13942), which would `KeyError` on a foreign name.

**Phase 2:** when `spec.tune_hook` is set (i.e. the project opts into the
portable seam), treat `(optimizer, model)` as opaque pass-through labels —
validate them against the config's enabled lists (or skip the
`OPTIMIZERS`/`MODELS`/`OPT_CLASS` checks entirely) rather than the grokking
profile taxonomy.

---

### Summary

The **config + search-space + kernel + hook are all correct and individually
verified** against `compile.py`'s documented schema/contract. End-to-end tuning
is blocked only by the three `compile.py` portability gaps above (one seam that
needs extending to the timing/validation paths, one hardcoded source glob, and
one hardcoded optimizer/model taxonomy). Closing them is the Phase-2 portability
proof; this directory is the fixture that proof will run against.
